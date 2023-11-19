import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Load_Assign import *

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set random seeds
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Load Dataset
train_data, test_data = load_MNIST()

# Get client dataloaders
iid_train_loader = iid_Assign(train_data)
noniid_train_loader = non_iid_Assign(train_data)

test_inputs, test_labels = zip(*test_data)
test_inputs = torch.stack(test_inputs).to(device)
test_labels = torch.Tensor(test_labels).to(device)
test_labels_np = test_labels.cpu().numpy()

# Model Defination
# Model 1: A simple multilayer-perceptron with 2-hidden
# layers with 200 units each using ReLu activations (199,210
# total parameters), which we refer to as the MNIST 2NN.
# define fully connected NN
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 200);
        self.fc2 = nn.Linear(200, 200);
        self.out = nn.Linear(200, 10);

    def forward(self, x):
        x = x.flatten(1) # torch.Size([B,784])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Fed_Avg
def fedavg(global_model, C, K, E, c_loader, rounds, lr, acc_threshold):
    # C is the fraction of clients that perform computation on each round.
    assert C <= 1 and C >= 0
    # K is the number of clients.
    # E is the local epochs, the number of training passes each client makes
    # over its local dataset on each round.
    
    c_per_round = max( round(C * K), 1)
    entropys_of_rounds = []
    VR_of_rounds = []
    accs_of_rounds = []
    predprob_of_rounds = []
    ece_of_rounds = []
    
    for i in range(rounds):
        
        # Choose c_per_round clients
        clients = random.sample(range(K), c_per_round)
        
        # Train on clients one by one
        client_model = []
        entropys_of_clients = []
        VR_of_clients = []
        for _, c in enumerate(clients):
            local_model, entropy, VR = client_training(global_model, id = c, local_epochs = E, dataloader = c_loader[c], lr = lr)
            local_model = local_model.to(device)
            client_model.append(local_model.state_dict())
            entropys_of_clients.append(entropy)
            VR_of_clients.append(VR)
        
        entropys_of_rounds.append(entropys_of_clients)
        VR_of_rounds.append(VR_of_clients)
    
        # Average and iterate global model parameters for next round  
        keys = client_model[0].keys()
        next_global_dict = {key: 0 for key in keys}
        for key in keys:
            values = [d[key] for d in client_model]
            averaged_value = sum(values) / len(values)
            next_global_dict[key] = averaged_value
            
        global_model.load_state_dict(next_global_dict)
        global_model = global_model.to(device)
        
        # Validate binned accuracy and ece this round
        ## Collect predicted probability (the maximum value from the softmax output)
        global_model.eval()
        with torch.no_grad():
            predictions = global_model(test_inputs)
        
        probabilities = F.softmax(predictions, dim = 1)
        max_probs, pred_c = torch.max(probabilities, dim=1)
        pred_probs = max_probs.cpu().numpy() # a numpy vector with length = N(test data no.)
        pred_labels = pred_c.cpu().numpy() # a numpy vector with length = N(test data no.)
        
        ## Sort bin and calculate calibration scores
        accs, predprob, ece = bin_Calculate_predprob_accs_ece(pred_probs, pred_labels, 
                                                   test_labels_np, bins = 20)
        
        accs_of_rounds.append(accs)
        predprob_of_rounds.append(predprob)
        ece_of_rounds.append(ece)

        # Calculate overall accuracy this round.
        predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
        true_labels = test_labels.cpu().numpy()
        acc_this_round = np.sum(true_labels == predicted_labels) / true_labels.shape[0]
        if acc_this_round >= acc_threshold:
            break
        
    return accs_of_rounds, predprob_of_rounds, ece_of_rounds, entropys_of_rounds, VR_of_rounds


# Calculate the averaged entropy of the local model after every local training
def calculate_entropy(output):
    # The differences between 2NN and CNN, 2NN doesn't have a softmax layer
    probabilities = F.softmax(output, dim=1)
    log_probabilities = torch.log(probabilities)
    entropy = -torch.sum(probabilities * log_probabilities, dim=1)
    return entropy.mean().item()
# Calculate Variation Ratio of the local model after every local training 
def calculate_VR(output):
    probabilities = F.softmax(output, dim=1)
    highest_prob, _ = torch.max(probabilities, dim=1)
    variation_ratio = 1 - highest_prob
    return variation_ratio.mean().item()

def bin_Calculate_predprob_accs_ece(probabilities, pred_labels, test_labels, bins=10):

    probabilities = np.where(probabilities == 0, 1e-8, probabilities)
    counts, bin_edges = np.histogram(probabilities, bins=bins, range=[0., 1.])
    indices = np.digitize(probabilities, bin_edges, right=True)
    accs = []
    predprob = []
    
    for b in range(1, 1 + bins):
        right_pred = 0
        num_thisbin = 0
        predprob_thisbin = []
        for i in range(len(indices)):
            if indices[i] == b:
                num_thisbin += 1
                predprob_thisbin.append(probabilities[i])
                if pred_labels[i] == test_labels[i]:
                    right_pred += 1
        if num_thisbin == 0:
            acc = 0
        else:
            acc = right_pred/num_thisbin
        
        if len(predprob_thisbin) == 0:
            predprob_mean = 0
        else:
            predprob_mean = sum(predprob_thisbin) / len(predprob_thisbin)
        
        accs.append(acc)
        predprob.append(predprob_mean)
    

    num_examples = np.sum(counts)
    ece = np.sum([(counts[i] / float(num_examples)) * np.sum(
        np.abs(predprob[i] - accs[i]))
                  for i in range(len(predprob)) if counts[i] > 0])
            
    return accs, predprob, ece

criterion = nn.CrossEntropyLoss()
def client_training(global_model, id, local_epochs, dataloader, lr):
    local_model = copy.deepcopy(global_model)
    local_model = local_model.to(device)
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
    
    for _ in range(local_epochs):
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # Calculate the Averaged Entropy After All Epoches:
    with torch.no_grad():
        test_outputs = local_model(test_inputs)
        entropy = calculate_entropy(test_outputs)
        VR = calculate_VR(test_outputs)
            
    return local_model, entropy, VR


# # MNIST 2NN
mlp = MLP()
acc_threshold_2nn = 0.97
