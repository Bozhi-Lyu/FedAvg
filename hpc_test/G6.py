from FedAvg_2NN import *
import sys

filename = "G6.txt"
sys.stdout = open(filename, 'w')
mlp_noniid6 = copy.deepcopy(mlp)
acc_mlp_noniid6, predprob_2nn_noniid6, ece_2nn_noniid6, entropy_2nn_noniid6, VR_2nn_noniid6 = fedavg(mlp_noniid6, C = 0.2, K = 100, E = 1, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid6")
print(acc_mlp_noniid6)

print("predprob_2nn_noniid6")
print(predprob_2nn_noniid6)

print("ece_2nn_noniid6")
print(ece_2nn_noniid6)

print("entropy_2nn_noniid6")
print(entropy_2nn_noniid6)

print("VR_2nn_noniid6")
print(VR_2nn_noniid6)

sys.stdout.close()