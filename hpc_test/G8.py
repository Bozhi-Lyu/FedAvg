from FedAvg_2NN import *
import sys

filename = "G8.txt"
sys.stdout = open(filename, 'w')
mlp_noniid8 = copy.deepcopy(mlp)
acc_mlp_noniid8, predprob_2nn_noniid8, ece_2nn_noniid8, entropy_2nn_noniid8, VR_2nn_noniid8 = fedavg(mlp_noniid8, C = 1, K = 100, E = 1, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid8")
print(acc_mlp_noniid8)

print("predprob_2nn_noniid8")
print(predprob_2nn_noniid8)

print("ece_2nn_noniid8")
print(ece_2nn_noniid8)

print("entropy_2nn_noniid8")
print(entropy_2nn_noniid8)

print("VR_2nn_noniid8")
print(VR_2nn_noniid8)

sys.stdout.close()