from FedAvg_2NN import *
import sys

filename = "G15.txt"
sys.stdout = open(filename, 'w')
mlp_noniid15 = copy.deepcopy(mlp)
acc_mlp_noniid15, predprob_2nn_noniid15, ece_2nn_noniid15, entropy_2nn_noniid15, VR_2nn_noniid15 = fedavg(mlp_noniid15, C = 0.5, K = 100, E = 20, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid15")
print(acc_mlp_noniid15)

print("predprob_2nn_noniid15")
print(predprob_2nn_noniid15)

print("ece_2nn_noniid15")
print(ece_2nn_noniid15)

print("entropy_2nn_noniid15")
print(entropy_2nn_noniid15)

print("VR_2nn_noniid15")
print(VR_2nn_noniid15)

sys.stdout.close()