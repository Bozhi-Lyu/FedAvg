from FedAvg_2NN import *
import sys
'''
overall_acc: oacc_i19
accs_bins_of_rounds: accs_bins_i19
predprob_of_rounds: pp_i19
ece_of_rounds: ece_i19
entropys_of_rounds: entro_i19
entrophys_bins_of_rounds: entro_bins_i19
VR_of_rounds: VR_i19
'''

filename = "G19.txt"
sys.stdout = open(filename, 'w')
mlp_iid19 = copy.deepcopy(mlp)
a1, a2, a3, a4, a5, a6, a7 = fedavg(mlp_iid19, C = 0.5, K = 100, E = 10, 
                      c_loader = iid_train_loader, rounds = 100, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("oacc_i19")
print(a1)

print("accs_bins_i19")
print(a2)

print("pp_i19")
print(a3)

print("ece_i19")
print(a4)

print("entro_i19")
print(a5)

print("entro_bins_i19")
print(a6)

print("VR_i19")
print(a7)

sys.stdout.close()