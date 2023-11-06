import random
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Data Load
def load_MNIST():
    # Load MNIST and return train and test.
    transform = transforms.Compose([transforms.ToTensor()])
    MNIST_train = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
    MNIST_test = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)

    return MNIST_train, MNIST_test

# Data Assign
def iid_Assign(dataset, batch_size = 10, n_clients = 100, n_per_client = 600):
    # Default: Partitioned into 100 clients, each receiving 600 examples.
    # Return a list of dataloaders for each client.

    split_datasets = torch.utils.data.random_split(dataset, [n_per_client] * n_clients)
    dataloaders = []
    for subset in split_datasets:
        dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
        dataloaders.append(dataloader)
    return dataloaders

def non_iid_Assign(dataset, batch_size = 10, n_clients = 100, n_per_client = 2):
    # "Non-IID, where we first sort the data by digit label, divide it into 200 
    # shards of size 300, and assign each of 100 clients 2 shards. This is a 
    # pathological non-IID partition of the data, as most clients will only 
    # have examples of two digits."
    
    # Default: Partitioned into 100 clients, each receiving 2*300 examples.
    # Return a list of dataloaders for each client.
    
    sorted_dataset = sorted(dataset, key=lambda x: x[1])
    subset_size = len(sorted_dataset) // (n_clients * n_per_client) 
    
    subdatasets = []
    for i in range(n_clients * n_per_client):
        start_index = i * subset_size
        subdataset = sorted_dataset[start_index : start_index + subset_size]
        subdatasets.append(subdataset)
    
    # Randomly choose n_per_client(2) shards for a client.
    random.shuffle(subdatasets)
    Shards = [subdatasets[i] + subdatasets[i+1] for i in range(0, len(subdatasets), n_per_client)]

    dataloaders = []
    for subset in Shards:
        dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
        dataloaders.append(dataloader)

    return dataloaders

def num_params(model):
    return sum(p.numel() for p in model.parameters())

# Visualization
def pltAcc(Acc, threshold, title):
    rounds = range(1, len(Acc) + 1)
    plt.plot(rounds, Acc, marker='.', linestyle='-', color='blue')
    plt.axhline(y=threshold, color='red', linestyle='--')
    plt.xlabel('Round')
    plt.ylabel('Acc')
    plt.title(title)
    
    last_round = len(rounds)
    last_acc = Acc[-1]
    annotation_text = f"round = {last_round}, acc = {last_acc}"
    plt.annotate(annotation_text, xy=(last_round, last_acc), xytext=(last_round, last_acc + 0.02),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.show()

def pltUncertainty(A, title):
    max_values = [np.max(a_i) for a_i in A]
    min_values = [np.min(a_i) for a_i in A]
    avg_values = [np.mean(a_i) for a_i in A]

    fig, ax = plt.subplots()

    ax.fill_between(range(len(A)), min_values, max_values, facecolor='red', alpha=0.2)

    ax.plot(range(len(A)), avg_values, color='blue')

    ax.set_xlabel('Rounds')
    ax.set_ylabel('Uncertainty(Entropy)')
    ax.set_title(title)
    ax.legend()

    plt.show()
    