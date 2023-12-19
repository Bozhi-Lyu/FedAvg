from FedAvg_2NN import *
import sys
'''
overall_acc: oacc_i21
accs_bins_of_rounds: accs_bins_i21
predprob_of_rounds: pp_i21
ece_of_rounds: ece_i21
entropys_of_rounds: entro_i21
entrophys_bins_of_rounds: entro_bins_i21
VR_of_rounds: VR_i21
'''

filename = "G21.txt"
sys.stdout = open(filename, 'w')
mlp_iid21 = copy.deepcopy(mlp)
a1, a2, a3, a4, a5, a6, a7 = fedavg(mlp_iid21, C = 0.1, K = 100, E = 20, 
                      c_loader = iid_train_loader, rounds = 100, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("oacc_i21")
print(a1)

print("accs_bins_i21")
print(a2)

print("pp_i21")
print(a3)

print("ece_i21")
print(a4)

print("entro_i21")
print(a5)

print("entro_bins_i21")
print(a6)

print("VR_i21")
print(a7)

sys.stdout.close()