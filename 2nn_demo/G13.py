from FedAvg_2NN import *
import sys

filename = "G13.txt"
sys.stdout = open(filename, 'w')
mlp_noniid13 = copy.deepcopy(mlp)
a1, a2, a3, a4, a5, a6, a7 = fedavg(mlp_noniid13, C = 0.1, K = 100, E = 20, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("oacc_n13")
print(a1)

print("accs_bins_n13")
print(a2)

print("pp_n13")
print(a3)

print("ece_n13")
print(a4)

print("entro_n13")
print(a5)

print("entro_bins_n13")
print(a6)

print("VR_n13")
print(a7)

sys.stdout.close()