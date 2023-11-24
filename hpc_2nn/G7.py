from FedAvg_2NN import *
import sys

'''
overall_acc: oacc_n5
accs_bins_of_rounds: accs_bins_n5
predprob_of_rounds: pp_n5
ece_of_rounds: ece_n5
entropys_of_rounds: entro_n5
entrophys_bins_of_rounds: entro_bins_n5
VR_of_rounds: VR_n5
'''

filename = "G7.txt"
sys.stdout = open(filename, 'w')
mlp_noniid7 = copy.deepcopy(mlp)
a1, a2, a3, a4, a5, a6, a7 = fedavg(mlp_noniid7, C = 0.5, K = 100, E = 1, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("oacc_n7")
print(a1)

print("accs_bins_n7")
print(a2)

print("pp_n7")
print(a3)

print("ece_n7")
print(a4)

print("entro_n7")
print(a5)

print("entro_bins_n7")
print(a6)

print("VR_n7")
print(a7)


sys.stdout.close()