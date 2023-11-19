from FedAvg_2NN import *
import sys

filename = "G16.txt"
sys.stdout = open(filename, 'w')
mlp_noniid16 = copy.deepcopy(mlp)
acc_mlp_noniid16, predprob_2nn_noniid16, ece_2nn_noniid16, entropy_2nn_noniid16, VR_2nn_noniid16 = fedavg(mlp_noniid16, C = 0.1, K = 100, E = 20, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid16")
print(acc_mlp_noniid16)

print("predprob_2nn_noniid16")
print(predprob_2nn_noniid16)

print("ece_2nn_noniid16")
print(ece_2nn_noniid16)

print("entropy_2nn_noniid16")
print(entropy_2nn_noniid16)

print("VR_2nn_noniid16")
print(VR_2nn_noniid16)

sys.stdout.close()