from FedAvg_2NN import *
import sys

filename = "G9.txt"
sys.stdout = open(filename, 'w')
mlp_noniid9 = copy.deepcopy(mlp)
acc_mlp_noniid9, predprob_2nn_noniid9, ece_2nn_noniid9, entropy_2nn_noniid9, VR_2nn_noniid9 = fedavg(mlp_noniid9, C = 0.1, K = 100, E = 10, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid9")
print(acc_mlp_noniid9)

print("predprob_2nn_noniid9")
print(predprob_2nn_noniid9)

print("ece_2nn_noniid9")
print(ece_2nn_noniid9)

print("entropy_2nn_noniid9")
print(entropy_2nn_noniid9)

print("VR_2nn_noniid9")
print(VR_2nn_noniid9)

sys.stdout.close()