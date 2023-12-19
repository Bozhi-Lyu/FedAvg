from FedAvg_CNN import *
import sys

# overall_acc: oacc_i15
# accs_bins_of_rounds: accs_bins_i15
# predprob_of_rounds: pp_i15
# ece_of_rounds: ece_i15
# entropys_of_rounds: entro_i15
# entrophys_bins_of_rounds: entro_bins_i15
# VR_of_rounds: VR_i15
# oe_of_rounds: oe_i15
# counts_of_rounds: counts_i15

filename = "G15.txt"
sys.stdout = open(filename, 'w')

GroupNum = [10, 11, 12, 13, 14, 15, 16, 17, 18]
C = [0.1, 0.5, 1, 0.1, 0.5, 1, 0.1, 0.5, 1]
E = [1, 1, 1, 10, 10, 10, 20, 20, 20]

cnn_noniid15 = copy.deepcopy(cnn)
a1, a2, a3, a4, a5, a6, a7, a8, a9  = fedavg(cnn_noniid15, 
                                    C = C[15 - 9 - 1], 
                                    K = 100, 
                                    E = E[15 - 9 - 1], 
                                    c_loader = noniid_train_loader, 
                                    rounds = 2000, 
                                    lr = 0.05, 
                                    acc_threshold = acc_threshold_cnn)

print("oacc_n15")
print(a1)

print("accs_bins_n15")
print(a2)

print("pp_n15")
print(a3)

print("ece_n15")
print(a4)

print("entro_n15")
print(a5)

print("entro_bins_n15")
print(a6)

print("VR_n15")
print(a7)

print("oe_n15")
print(a8)

print("counts_n15")
print(a9)

sys.stdout.close()
