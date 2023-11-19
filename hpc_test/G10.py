from FedAvg_2NN import *
import sys

filename = "G10.txt"
sys.stdout = open(filename, 'w')
mlp_noniid10 = copy.deepcopy(mlp)
acc_mlp_noniid10, predprob_2nn_noniid10, ece_2nn_noniid10, entropy_2nn_noniid10, VR_2nn_noniid10 = fedavg(mlp_noniid10, C = 0.2, K = 100, E = 10, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid10")
print(acc_mlp_noniid10)

print("predprob_2nn_noniid10")
print(predprob_2nn_noniid10)

print("ece_2nn_noniid10")
print(ece_2nn_noniid10)

print("entropy_2nn_noniid10")
print(entropy_2nn_noniid10)

print("VR_2nn_noniid10")
print(VR_2nn_noniid10)

sys.stdout.close()