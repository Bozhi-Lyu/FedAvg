from FedAvg_CNN import *
import sys

# overall_acc: oacc_i10
# accs_bins_of_rounds: accs_bins_i10
# predprob_of_rounds: pp_i10
# ece_of_rounds: ece_i10
# entropys_of_rounds: entro_i10
# entrophys_bins_of_rounds: entro_bins_i10
# VR_of_rounds: VR_i10


filename = "G10.txt"
sys.stdout = open(filename, 'w')

GroupNum = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
C = [0.1, 0.2, 0.5, 1, 0.1, 0.2, 0.5, 1, 0.1, 0.2, 0.5, 1]
E = [1, 1, 1, 1, 10, 10, 10, 10, 20, 20, 20, 20]

cnn_iid10 = copy.deepcopy(cnn)
a1, a2, a3, a4, a5, a6, a7 = fedavg(cnn_iid10, 
                                    C = C[10 - 1], 
                                    K = 100, 
                                    E = E[10 - 1], 
                                    c_loader = iid_train_loader, 
                                    rounds = 1200, 
                                    lr = 0.05, 
                                    acc_threshold = acc_threshold_cnn)

print("oacc_i10")
print(a1)

print("accs_bins_i10")
print(a2)

print("pp_i10")
print(a3)

print("ece_i10")
print(a4)

print("entro_i10")
print(a5)

print("entro_bins_i10")
print(a6)

print("VR_i10")
print(a7)

sys.stdout.close()
