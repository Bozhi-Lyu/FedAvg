from FedAvg_2NN import *
import sys

filename = "G12.txt"
sys.stdout = open(filename, 'w')
mlp_noniid12 = copy.deepcopy(mlp)
acc_mlp_noniid12, predprob_2nn_noniid12, ece_2nn_noniid12, entropy_2nn_noniid12, VR_2nn_noniid12 = fedavg(mlp_noniid12, C = 1, K = 100, E = 10, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid12")
print(acc_mlp_noniid12)

print("predprob_2nn_noniid12")
print(predprob_2nn_noniid12)

print("ece_2nn_noniid12")
print(ece_2nn_noniid12)

print("entropy_2nn_noniid12")
print(entropy_2nn_noniid12)

print("VR_2nn_noniid12")
print(VR_2nn_noniid12)

sys.stdout.close()