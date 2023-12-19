import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator


import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns

namelist = ["IID, C = 0.1, K = 100, E = 1",#0-#3. G1-G4
            "IID, C = 0.2, K = 100, E = 1",
            "IID, C = 0.5, K = 100, E = 1",
            "IID, C = 1, K = 100, E = 1",

            "IID, C = 0.1, K = 100, E = 10",#4-#7, G5-G8
            "IID, C = 0.2, K = 100, E = 10",
            "IID, C = 0.5, K = 100, E = 10",
            "IID, C = 1, K = 100, E = 10", 

            "IID, C = 0.1, K = 100, E = 20",#8-#11, G9-G12
            "IID, C = 0.2, K = 100, E = 20",
            "IID, C = 0.5, K = 100, E = 20",
            "IID, C = 1, K = 100, E = 20",

            "non-IID, C = 0.1, K = 100, E = 1", #16-#19, G17-G20
            "non-IID, C = 0.2, K = 100, E = 1", 
            "non-IID, C = 0.5, K = 100, E = 1", 
            "non-IID, C = 1, K = 100, E = 1", 

            "non-IID, C = 0.1, K = 100, E = 10", #20-#23, G21-G24
            "non-IID, C = 0.2, K = 100, E = 10", 
            "non-IID, C = 0.5, K = 100, E = 10", 
            "non-IID, C = 1, K = 100, E = 10", 

            "non-IID, C = 0.1, K = 100, E = 20", #12-#15, G13-G16
            "non-IID, C = 0.2, K = 100, E = 20", 
            "non-IID, C = 0.5, K = 100, E = 20", 
            "non-IID, C = 1, K = 100, E = 20"]

colorlist = ["red", "brown", "blue", "green"]

def get_unique_labels(dataloaders):
    unique_labels = set()
    
    for dataloader in dataloaders:
        for _, labels in dataloader:
            unique_labels.update(set(labels.numpy()))
    
    return unique_labels

def PlotDataDistribution(iid_train_loader, noniid_train_loader):
    set1 = get_unique_labels(iid_train_loader)
    set2 = get_unique_labels(noniid_train_loader)
    assert( set1 == set2 )
    classnum = len(set1)
    
    iid_labels = []
    for i in iid_train_loader:
        iid_label = torch.zeros(classnum)
        for (x,y) in i:
            iid_label += torch.sum(F.one_hot(y, num_classes=classnum), dim=0)
        iid_labels.append(iid_label)
        
    iid_labels = torch.stack(iid_labels)
    iid_normalized = iid_labels / 600 # min-max normalization
    
    # non_iid
    noniid_labels = []
    for i in noniid_train_loader:
        noniid_label = torch.zeros(classnum)
        for (x,y) in i:
            noniid_label += torch.sum(F.one_hot(y, num_classes=classnum), dim=0)
        noniid_labels.append(noniid_label)
        
    noniid_labels = torch.stack(noniid_labels)
    noniid_normalized = noniid_labels / 600
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(iid_normalized.numpy(), cmap='Blues', fmt=".2f", cbar=True, annot=False)
    plt.yticks(np.arange(0, 100, step=10), [i * 10 for i in range(10)])
    plt.xlabel('Classes')
    plt.ylabel('Clients No.')
    plt.title('IID Distribution Heatmap')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(noniid_normalized.numpy(), cmap='Blues', fmt=".2f", cbar=True, annot=False)
    plt.yticks(np.arange(0, 100, step=10), [i * 10 for i in range(10)])
    plt.xlabel('Classes')
    plt.ylabel('Clients No.')
    plt.title('Non-IID Distribution Heatmap')
    
    plt.tight_layout()
    plt.show()

def PlotAcc(acclists, labels, iidno, noniidno, epochs = [1] * len(namelist)):
    
    assert( len(acclists) == len(labels) )
    assert( len(acclists) == iidno + noniidno )
    
    fig, ax = plt.subplots(figsize=(18, 12))
    cmap = cm.get_cmap('gist_rainbow')

    for i in range(len(acclists)):
        no = list(range(1, len(acclists[i]) + 1))
        x = [j * epochs[i] for j in no]
        y = acclists[i]
        
        color = cmap(i / (len(acclists) - 1))
        
        if i < iidno:
            linestyle = '--'
        else:
            linestyle = '-'
        ax.plot(x, y, label = labels[i], linewidth=1, 
                linestyle = linestyle, color = color)
    
    ax.axhline(y=0.99, color='gray', linestyle='--')
    ax.text(0, 0.99, 'Accuracy Threshold = 0.99', ha='left')
    
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.9, 0.995)
    ax.legend(loc='lower right')
    plt.show()
    
def PlotUncertaintyVSAcc(binEntro_of_rounds, binAcc_of_rounds, label):
    assert len(binEntro_of_rounds) == len(binAcc_of_rounds), "Inconsistent list length"

    fig, ax = plt.subplots(figsize=(18, 12))
    cmap = matplotlib.colormaps.get_cmap('turbo')

    for i in range(len(binEntro_of_rounds)):
        assert len(binEntro_of_rounds[i]) == len(binAcc_of_rounds[i]), "Inconsistent list length"
        color = cmap(i / (len(binEntro_of_rounds) - 1))
        ax.scatter(binAcc_of_rounds[i], binEntro_of_rounds[i], color=color)

    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy VS Accuracy' + '\n' + label)
    
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array(list(range(len(binEntro_of_rounds))))
    cbar = plt.colorbar(sm, ax=ax) 
    num_ticks = len(binEntro_of_rounds)
    cbar.set_ticks(range(num_ticks))
    cbar.set_label('Training Rounds')
    tick_labels = [str(i+1) for i in range(num_ticks)]
    cbar.ax.set_yticklabels(tick_labels)
    
    interval = 10
    cbar.ax.yaxis.set_major_locator(MultipleLocator(interval))

    plt.show()

def PlotPPVSAcc(binPPo_of_rounds, binAcc_of_rounds, label):
     assert len(binPPo_of_rounds) == len(binAcc_of_rounds), "Inconsistent list length"

     fig, ax = plt.subplots(figsize=(18, 12))
     cmap = matplotlib.colormaps.get_cmap('turbo')

     for i in range(len(binPPo_of_rounds)):
         assert len(binPPo_of_rounds[i]) == len(binAcc_of_rounds[i]), "Inconsistent list length"
         color = cmap(i / (len(binPPo_of_rounds) - 1))
         ax.scatter(binAcc_of_rounds[i], binPPo_of_rounds[i], color=color)
         
     ax.plot([0, 1], [0, 1], color='black', linestyle=':')
     ax.set_xlabel('Accuracy')
     ax.set_ylabel('Predicted Probability')
     ax.set_title('Predicted Probability VS Accuracy' + '\n' + label)
     
     sm = cm.ScalarMappable(cmap=cmap)
     sm.set_array(list(range(len(binPPo_of_rounds))))
     cbar = plt.colorbar(sm, ax=ax) 
     num_ticks = len(binPPo_of_rounds)
     cbar.set_ticks(range(num_ticks))
     cbar.set_label('Training Rounds')
     tick_labels = [str(i+1) for i in range(num_ticks)]
     cbar.ax.set_yticklabels(tick_labels)
     
     interval = 10
     cbar.ax.yaxis.set_major_locator(MultipleLocator(interval))

     plt.show()   

def PlotECEVSOAcc(values, indices, rounds):
    ecelist = [values[3 + 9 * (i - 1)] for i in indices]
    oacclist = [values[9 * (i - 1)] for i in indices]
    labels = ["G" + str(i) + ", " + namelist[i - 1] for i in indices]
    innerPlotECEVSOAcc(ecelist, oacclist, labels, rounds)

def innerPlotECEVSOAcc(ecelist, oacclist, labels, rounds):
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    ax = plt.subplot(gs[0])
    cmap = matplotlib.colormaps.get_cmap('brg')
    
    
    for i in range(len(ecelist)):
        if len(ecelist[i]) > rounds:
            x1 = range(1, rounds + 1)
            y1 = ecelist[i][:rounds]
        else: 
            x1 = range(1, len(ecelist[i]) + 1)
            y1 = ecelist[i]
        ax.plot(x1, y1, color=cmap(i / (len(ecelist) - 1)), 
                linestyle=':', label='ECE Score ' + labels[i])
        
    ax.set_xlabel('Rounds')
    ax.set_ylabel('ECE Score')
    ax.set_ylim(0, 0.6)
    ax.yaxis.label.set_color('blue')
    ax.tick_params(axis='y', colors='blue')
    ax.spines['left'].set_color('blue')
    ax.spines['right'].set_color('red')
    
    ax2 = ax.twinx()
    for i in range(len(oacclist)):
        if len(oacclist[i]) > rounds:
            x2 = range(1, rounds + 1)
            y2 = oacclist[i][:rounds]
        else:
            x2 = range(1, len(oacclist[i]) + 1)
            y2 = oacclist[i]

        ax2.plot(x2, y2, color=cmap(i / (len(oacclist) - 1)), 
                linestyle='-', label='Accuracy ' + labels[i])
    ax2.set_ylabel('Accuracy')
    ax2.yaxis.label.set_color('red')
    ax2.tick_params(axis='y', colors='red')
    ax2.spines['right'].set_color('red')
    ax2.spines['left'].set_color('blue')
    ax2.set_ylim(0.6, 1)
    legend_ax = plt.subplot(gs[1])
    lines, labels = ax.get_legend_handles_labels()
    legend_ax.legend(lines, labels, loc='center')
    legend_ax.axis('off')  
    fig.set_size_inches(15, 6)
    
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
