from FedAvg_2NN import *
import sys

filename = "G14.txt"
sys.stdout = open(filename, 'w')
mlp_noniid14 = copy.deepcopy(mlp)
acc_mlp_noniid14, predprob_2nn_noniid14, ece_2nn_noniid14, entropy_2nn_noniid14, VR_2nn_noniid14 = fedavg(mlp_noniid14, C = 0.2, K = 100, E = 20, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid14")
print(acc_mlp_noniid14)

print("predprob_2nn_noniid14")
print(predprob_2nn_noniid14)

print("ece_2nn_noniid14")
print(ece_2nn_noniid14)

print("entropy_2nn_noniid14")
print(entropy_2nn_noniid14)

print("VR_2nn_noniid14")
print(VR_2nn_noniid14)

sys.stdout.close()