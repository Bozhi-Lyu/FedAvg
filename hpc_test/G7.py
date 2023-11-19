from FedAvg_2NN import *
import sys

filename = "G7.txt"
sys.stdout = open(filename, 'w')
mlp_noniid7 = copy.deepcopy(mlp)
acc_mlp_noniid7, predprob_2nn_noniid7, ece_2nn_noniid7, entropy_2nn_noniid7, VR_2nn_noniid7 = fedavg(mlp_noniid7, C = 0.5, K = 100, E = 1, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid7")
print(acc_mlp_noniid7)

print("predprob_2nn_noniid7")
print(predprob_2nn_noniid7)

print("ece_2nn_noniid7")
print(ece_2nn_noniid7)

print("entropy_2nn_noniid7")
print(entropy_2nn_noniid7)

print("VR_2nn_noniid7")
print(VR_2nn_noniid7)

sys.stdout.close()