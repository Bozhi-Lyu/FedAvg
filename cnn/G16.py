from FedAvg_CNN import *
import sys

# overall_acc: oacc_i16
# accs_bins_of_rounds: accs_bins_i16
# predprob_of_rounds: pp_i16
# ece_of_rounds: ece_i16
# entropys_of_rounds: entro_i16
# entrophys_bins_of_rounds: entro_bins_i16
# VR_of_rounds: VR_i16
# oe_of_rounds: oe_i16
# counts_of_rounds: counts_i16

filename = "G16.txt"
sys.stdout = open(filename, 'w')

GroupNum = [10, 11, 12, 13, 14, 15, 16, 17, 18]
C = [0.1, 0.5, 1, 0.1, 0.5, 1, 0.1, 0.5, 1]
E = [1, 1, 1, 10, 10, 10, 20, 20, 20]

cnn_noniid16 = copy.deepcopy(cnn)
a1, a2, a3, a4, a5, a6, a7, a8, a9  = fedavg(cnn_noniid16, 
                                    C = C[16 - 9 - 1], 
                                    K = 100, 
                                    E = E[16 - 9 - 1], 
                                    c_loader = noniid_train_loader, 
                                    rounds = 2000, 
                                    lr = 0.05, 
                                    acc_threshold = acc_threshold_cnn)

print("oacc_n16")
print(a1)

print("accs_bins_n16")
print(a2)

print("pp_n16")
print(a3)

print("ece_n16")
print(a4)

print("entro_n16")
print(a5)

print("entro_bins_n16")
print(a6)

print("VR_n16")
print(a7)

print("oe_n16")
print(a8)

print("counts_n16")
print(a9)

sys.stdout.close()
