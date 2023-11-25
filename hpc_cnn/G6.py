from FedAvg_CNN import *
import sys

# overall_acc: oacc_i6
# accs_bins_of_rounds: accs_bins_i6
# predprob_of_rounds: pp_i6
# ece_of_rounds: ece_i6
# entropys_of_rounds: entro_i6
# entrophys_bins_of_rounds: entro_bins_i6
# VR_of_rounds: VR_i6


filename = "G6.txt"
sys.stdout = open(filename, 'w')

GroupNum = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
C = [0.1, 0.2, 0.5, 1, 0.1, 0.2, 0.5, 1, 0.1, 0.2, 0.5, 1]
E = [1, 1, 1, 1, 10, 10, 10, 10, 20, 20, 20, 20]

cnn_iid6 = copy.deepcopy(cnn)
a1, a2, a3, a4, a5, a6, a7 = fedavg(cnn_iid6, 
                                    C = C[6 - 1], 
                                    K = 100, 
                                    E = E[6 - 1], 
                                    c_loader = iid_train_loader, 
                                    rounds = 1200, 
                                    lr = 0.05, 
                                    acc_threshold = acc_threshold_cnn)

print("oacc_i6")
print(a1)

print("accs_bins_i6")
print(a2)

print("pp_i6")
print(a3)

print("ece_i6")
print(a4)

print("entro_i6")
print(a5)

print("entro_bins_i6")
print(a6)

print("VR_i6")
print(a7)

sys.stdout.close()
