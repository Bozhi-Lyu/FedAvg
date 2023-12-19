from FedAvg_2NN import *
import sys

filename = "G12.txt"
sys.stdout = open(filename, 'w')
mlp_noniid12 = copy.deepcopy(mlp)
a1, a2, a3, a4, a5, a6, a7 = fedavg(mlp_noniid12, C = 1, K = 100, E = 10, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("oacc_n12")
print(a1)

print("accs_bins_n12")
print(a2)

print("pp_n12")
print(a3)

print("ece_n12")
print(a4)

print("entro_n12")
print(a5)

print("entro_bins_n12")
print(a6)

print("VR_n12")
print(a7)

sys.stdout.close()