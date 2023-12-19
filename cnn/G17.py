from FedAvg_CNN import *
import sys

# overall_acc: oacc_i17
# accs_bins_of_rounds: accs_bins_i17
# predprob_of_rounds: pp_i17
# ece_of_rounds: ece_i17
# entropys_of_rounds: entro_i17
# entrophys_bins_of_rounds: entro_bins_i17
# VR_of_rounds: VR_i17
# oe_of_rounds: oe_i17
# counts_of_rounds: counts_i17

filename = "G17.txt"
sys.stdout = open(filename, 'w')

GroupNum = [10, 11, 12, 13, 14, 15, 16, 17, 18]
C = [0.1, 0.5, 1, 0.1, 0.5, 1, 0.1, 0.5, 1]
E = [1, 1, 1, 10, 10, 10, 20, 20, 20]

cnn_noniid17 = copy.deepcopy(cnn)
a1, a2, a3, a4, a5, a6, a7, a8, a9  = fedavg(cnn_noniid17, 
                                    C = C[17 - 9 - 1], 
                                    K = 100, 
                                    E = E[17 - 9 - 1], 
                                    c_loader = noniid_train_loader, 
                                    rounds = 2000, 
                                    lr = 0.05, 
                                    acc_threshold = acc_threshold_cnn)

print("oacc_n17")
print(a1)

print("accs_bins_n17")
print(a2)

print("pp_n17")
print(a3)

print("ece_n17")
print(a4)

print("entro_n17")
print(a5)

print("entro_bins_n17")
print(a6)

print("VR_n17")
print(a7)

print("oe_n17")
print(a8)

print("counts_n17")
print(a9)

sys.stdout.close()
