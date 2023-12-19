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

# print("Training data")
# x, y = next(iter(iid_train_loader[0]))
# print("Local Batch dimension (B x C x H x W):", x.shape)

test_inputs, test_labels = zip(*test_data)
test_inputs = torch.stack(test_inputs).to(device)
test_labels = torch.Tensor(test_labels).to(device)
test_labels_np = test_labels.cpu().numpy()

# Model Defination
# Model 1: A simple multilayer-perceptron with 2-hidden
# layers with 200 units each using ReLu activations (199,210
# total parameters), which we refer to as the MNIST CNN.
# define fully connected NN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print("After x = F.relu(self.conv1(x)), the shape of x is: ", x.shape)
        x = self.pool1(x)
        # print("After x = self.pool1(x), the shape of x is: ", x.shape)
        x = F.relu(self.conv2(x))
        # print("After x = F.relu(self.conv2(x)), the shape of x is: ", x.shape)    
        x = self.pool2(x)
        # print("After x = self.pool1(F.relu(self.conv1(x))), the shape of x is: ", x.shape)
        
        x = x.view(-1, 64 * 7 * 7) # flatten
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
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
    entrophys_bins_of_rounds = []
    VR_of_rounds = []
    accs_bins_of_rounds = []
    predprob_of_rounds = []
    ece_of_rounds = []
    oe_of_rounds = []
    counts_of_rounds = []
    overall_acc = []
    
    for i in range(rounds):
        
        # Choose c_per_round clients
        clients = random.sample(range(K), c_per_round)
        
        # Train on clients one by one
        client_model = []
        entropys_of_clients = []
        VR_of_clients = []
        for _, c in enumerate(clients):
            local_model, entropy, VR = client_training(global_model, id = c, 
                                                       local_epochs = E, 
                                                       dataloader = c_loader[c], 
                                                       lr = lr)
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
            probabilities = global_model(test_inputs)
        ## Avoid situations where the log operation has a 0 base in subsequent calculations.
        probabilities = torch.where(probabilities == 0, torch.tensor(1e-8), probabilities)
        
        # max_probs, pred_c = torch.max(probabilities, dim=1)
        # pred_probs = max_probs.cpu().numpy() # a numpy vector with length = N(test data no.)
        # pred_labels = pred_c.cpu().numpy() # a numpy vector with length = N(test data no.)
        
        ## Sort bin and calculate calibration scores
        accs, predprob, counts, ece, oe, entrophy_inbins = bin_Calculate_predprob_accs_ece(probabilities, 
                                                   test_labels_np, bins = 200)
        
        accs_bins_of_rounds.append(accs)
        predprob_of_rounds.append(predprob)
        ece_of_rounds.append(ece)
        entrophys_bins_of_rounds.append(entrophy_inbins)
        oe_of_rounds.append(oe)
        counts_of_rounds.append(counts)

        # Calculate overall accuracy this round.
        predicted_labels = torch.argmax(probabilities, dim=1).cpu().numpy()
        true_labels = test_labels.cpu().numpy()
        acc_this_round = np.sum(true_labels == predicted_labels) / true_labels.shape[0]
        overall_acc.append(acc_this_round)
        
        # if meet the required acc
        if acc_this_round >= acc_threshold:
            break
        
    return overall_acc, accs_bins_of_rounds, predprob_of_rounds, ece_of_rounds, \
        entropys_of_rounds, entrophys_bins_of_rounds, VR_of_rounds, oe_of_rounds, counts_of_rounds


# Calculate the averaged entropy of the local model after every local training
def calculate_entropy(output):
    # The differences between 2NN and CNN, 2NN doesn't have a softmax layer
    log_probabilities = torch.log(output)
    entropy = -torch.sum(output * log_probabilities, dim=1)
    return entropy.mean().item()
# Calculate Variation Ratio of the local model after every local training 
def calculate_VR(output):

    highest_prob, _ = torch.max(output, dim=1)
    variation_ratio = 1 - highest_prob
    return variation_ratio.mean().item()

def bin_Calculate_predprob_accs_ece(probabilities, test_labels, bins=10):
    '''
    Parameters
    ----------
    probabilities : Tensor, softmax output, 0 replaced.
    test_labels : numpy vector, true labels
    bins : int, optional, the default is 10.

    Returns
    -------
    accs : list, length = bin no.
        averaged accuracy in bins.
    predprob : list, length = bin no.
        averaged predicted probabilities in bins.
    ece : float, ece on all bins.
    entrophys : list, length = bin no.
        averaged entrophy in bins.
    '''
    max_probs, pred_c = torch.max(probabilities, dim=1)
    pred_probs = max_probs.cpu().numpy() # a numpy vector with length = N(test data no.)
    pred_labels = pred_c.cpu().numpy() # a numpy vector with length = N(test data no.)

    counts, bin_edges = np.histogram(pred_probs, bins=bins, range=[0., 1.])
    indices = np.digitize(pred_probs, bin_edges, right=True) # Start from 1!!
    
    accs = [] # Accuracy of Bm, list with length = bins
    predprob = [] # Confidence of Bm, list with length = bins
    entrophys = []
    
    prob_bin = [None] * bins # Record the possibilities of samples.
    predlabel_bin = [None] * bins # Record the predicted labels.
    testlabel_bin = [None] * bins # Record the true labels.
    p_hat_bin = [None] * bins # Record the winning score(confidence) of samples.
    
    for i in range(len(pred_probs)):
        index = indices[i] - 1
        if prob_bin[index] is None:
            prob_bin[index] = probabilities[i, :].unsqueeze(0)
            predlabel_bin[index] = [pred_labels[i]]
            testlabel_bin[index] = [test_labels[i]]
            p_hat_bin[index] = [pred_probs[i]] 
        else:
            prob_bin[index] = torch.cat((prob_bin[index], 
                                         probabilities[i, :].unsqueeze(0)), dim=0)
            predlabel_bin[index].append(pred_labels[i])
            testlabel_bin[index].append(test_labels[i])
            p_hat_bin[index].append(pred_probs[i])

    for i in range(bins):
        
        if prob_bin[i] is None:
            predprob_mean = None
            acc = None
            ave_entro = None
        else:
            right_pred = sum(x == y for x, y in 
                 zip(predlabel_bin[i], testlabel_bin[i]))
            predprob_mean = sum(p_hat_bin[i]) / len(p_hat_bin[i])
            acc = right_pred/counts[i]
            log_probabilities = torch.log(prob_bin[i])
            entrophy_thisbin = -torch.sum(prob_bin[i] * log_probabilities, dim=1)
            ave_entro = entrophy_thisbin.mean().item()
            
        predprob.append(predprob_mean)
        accs.append(acc)
        entrophys.append(ave_entro)

    num_examples = np.sum(counts)
    assert num_examples == len(pred_probs), "Something Wrong..."
    for i in range(len(counts)):
        if counts[i] == 0:
            assert accs[i] is None, "counts, accs not match."
            assert predprob[i] is None, "counts, predprob not match."
            assert entrophys[i] is None, "counts, entrophys not match."

    
    ece = np.sum([(counts[i] / float(num_examples)) * np.abs(predprob[i] - accs[i])
                  for i in range(len(predprob)) if counts[i] > 0])
    oe = np.sum([(counts[i] / float(num_examples)) * ( predprob[i] * max(predprob[i] - accs[i], 0) )
                  for i in range(len(predprob)) if counts[i] > 0])

    return accs, predprob, counts.tolist(), ece, oe, entrophys


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
        test_outputs = torch.where(test_outputs == 0, torch.tensor(1e-8), test_outputs)
        entropy = calculate_entropy(test_outputs)
        VR = calculate_VR(test_outputs)
            
    return local_model, entropy, VR


# # MNIST CNN
cnn = CNN()
acc_threshold_cnn = 0.99
