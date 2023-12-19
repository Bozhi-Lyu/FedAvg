from FedAvg_CNN import *
import sys

# overall_acc: oacc_i12
# accs_bins_of_rounds: accs_bins_i12
# predprob_of_rounds: pp_i12
# ece_of_rounds: ece_i12
# entropys_of_rounds: entro_i12
# entrophys_bins_of_rounds: entro_bins_i12
# VR_of_rounds: VR_i12
# oe_of_rounds: oe_i12
# counts_of_rounds: counts_i12

filename = "G12.txt"
sys.stdout = open(filename, 'w')

GroupNum = [10, 11, 12, 13, 14, 15, 16, 17, 18]
C = [0.1, 0.5, 1, 0.1, 0.5, 1, 0.1, 0.5, 1]
E = [1, 1, 1, 10, 10, 10, 20, 20, 20]

cnn_noniid12 = copy.deepcopy(cnn)
a1, a2, a3, a4, a5, a6, a7, a8, a9  = fedavg(cnn_noniid12, 
                                    C = C[12 - 9 - 1], 
                                    K = 100, 
                                    E = E[12 - 9 - 1], 
                                    c_loader = noniid_train_loader, 
                                    rounds = 2000, 
                                    lr = 0.05, 
                                    acc_threshold = acc_threshold_cnn)

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

print("oe_n12")
print(a8)

print("counts_n12")
print(a9)

sys.stdout.close()
