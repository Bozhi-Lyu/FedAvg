from FedAvg_2NN import *
import sys

filename = "G11.txt"
sys.stdout = open(filename, 'w')
mlp_noniid11 = copy.deepcopy(mlp)
acc_mlp_noniid11, predprob_2nn_noniid11, ece_2nn_noniid11, entropy_2nn_noniid11, VR_2nn_noniid11 = fedavg(mlp_noniid11, C = 0.5, K = 100, E = 10, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid11")
print(acc_mlp_noniid11)

print("predprob_2nn_noniid11")
print(predprob_2nn_noniid11)

print("ece_2nn_noniid11")
print(ece_2nn_noniid11)

print("entropy_2nn_noniid11")
print(entropy_2nn_noniid11)

print("VR_2nn_noniid11")
print(VR_2nn_noniid11)

sys.stdout.close()