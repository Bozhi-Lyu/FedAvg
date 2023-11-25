import random
import torch
import torchvision
import torchvision.transforms as transforms

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