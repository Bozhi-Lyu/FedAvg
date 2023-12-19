from FedAvg_2NN import *
import sys

'''
overall_acc: oacc_n11
accs_bins_of_rounds: accs_bins_n11
predprob_of_rounds: pp_n11
ece_of_rounds: ece_n11
entropys_of_rounds: entro_n11
entrophys_bins_of_rounds: entro_bins_n11
VR_of_rounds: VR_n11
'''

filename = "G11.txt"
sys.stdout = open(filename, 'w')
mlp_noniid11 = copy.deepcopy(mlp)
a1, a2, a3, a4, a5, a6, a7 = fedavg(mlp_noniid11, C = 0.5, K = 100, E = 10, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("oacc_n11")
print(a1)

print("accs_bins_n11")
print(a2)

print("pp_n11")
print(a3)

print("ece_n11")
print(a4)

print("entro_n11")
print(a5)

print("entro_bins_n11")
print(a6)

print("VR_n11")
print(a7)


sys.stdout.close()