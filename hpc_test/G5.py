from FedAvg_2NN import *
import sys

filename = "G5.txt"
sys.stdout = open(filename, 'w')
mlp_noniid5 = copy.deepcopy(mlp)
acc_mlp_noniid5, predprob_2nn_noniid5, ece_2nn_noniid5, entropy_2nn_noniid5, VR_2nn_noniid5 = fedavg(mlp_noniid5, C = 0.1, K = 100, E = 1, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid5")
print(acc_mlp_noniid5)

print("predprob_2nn_noniid5")
print(predprob_2nn_noniid5)

print("ece_2nn_noniid5")
print(ece_2nn_noniid5)

print("entropy_2nn_noniid5")
print(entropy_2nn_noniid5)

print("VR_2nn_noniid5")
print(VR_2nn_noniid5)

sys.stdout.close()