from FedAvg_2NN import *
import sys

filename = "G13.txt"
sys.stdout = open(filename, 'w')
mlp_noniid13 = copy.deepcopy(mlp)
acc_mlp_noniid13, predprob_2nn_noniid13, ece_2nn_noniid13, entropy_2nn_noniid13, VR_2nn_noniid13 = fedavg(mlp_noniid13, C = 0.1, K = 100, E = 20, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid13")
print(acc_mlp_noniid13)

print("predprob_2nn_noniid13")
print(predprob_2nn_noniid13)

print("ece_2nn_noniid13")
print(ece_2nn_noniid13)

print("entropy_2nn_noniid13")
print(entropy_2nn_noniid13)

print("VR_2nn_noniid13")
print(VR_2nn_noniid13)

sys.stdout.close()