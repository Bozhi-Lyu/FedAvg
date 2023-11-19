from FedAvg_2NN import *
import sys

filename = "G1.txt"
sys.stdout = open(filename, 'w')
mlp_iid1 = copy.deepcopy(mlp)
acc_mlp_iid1, predprob_2nn_iid1, ece_2nn_iid1, entropy_2nn_iid1, VR_2nn_iid1 = fedavg(mlp_iid1, C = 0.1, K = 100, E = 1, 
                      c_loader = iid_train_loader, rounds = 100, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_iid1")
print(acc_mlp_iid1)

print("predprob_2nn_iid1")
print(predprob_2nn_iid1)

print("ece_2nn_iid1")
print(ece_2nn_iid1)

print("entropy_2nn_iid1")
print(entropy_2nn_iid1)

print("VR_2nn_iid1")
print(VR_2nn_iid1)



sys.stdout.close()

filename = "G5.txt"
sys.stdout = open(filename, 'w')
mlp_noniid5 = copy.deepcopy(mlp)
acc_mlp_noniid5, predprob_2nn_noniid5, ece_2nn_noniid5, entropy_2nn_noniid5, VR_2nn_noniid5 = fedavg(mlp_noniid5, C = 0.1, K = 100, E = 1, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid5")
print(acc_mlp_noniid5)

print("predprob_2nn_noniid5")
print(predprob_2nn_noniid5)

print("ece_2nn_noniid5")
print(ece_2nn_noniid5)

print("entropy_2nn_noniid5")
print(entropy_2nn_noniid5)

print("VR_2nn_noniid5")
print(VR_2nn_noniid5)

sys.stdout.close()


filename = "G9.txt"
sys.stdout = open(filename, 'w')
mlp_noniid9 = copy.deepcopy(mlp)
acc_mlp_noniid9, predprob_2nn_noniid9, ece_2nn_noniid9, entropy_2nn_noniid9, VR_2nn_noniid9 = fedavg(mlp_noniid9, C = 0.1, K = 100, E = 10, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid9")
print(acc_mlp_noniid9)

print("predprob_2nn_noniid9")
print(predprob_2nn_noniid9)

print("ece_2nn_noniid9")
print(ece_2nn_noniid9)

print("entropy_2nn_noniid9")
print(entropy_2nn_noniid9)

print("VR_2nn_noniid9")
print(VR_2nn_noniid9)

sys.stdout.close()


filename = "G13.txt"
sys.stdout = open(filename, 'w')
mlp_noniid13 = copy.deepcopy(mlp)
acc_mlp_noniid13, predprob_2nn_noniid13, ece_2nn_noniid13, entropy_2nn_noniid13, VR_2nn_noniid13 = fedavg(mlp_noniid13, C = 0.1, K = 100, E = 20, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid13")
print(acc_mlp_noniid13)

print("predprob_2nn_noniid13")
print(predprob_2nn_noniid13)

print("ece_2nn_noniid13")
print(ece_2nn_noniid13)

print("entropy_2nn_noniid13")
print(entropy_2nn_noniid13)

print("VR_2nn_noniid13")
print(VR_2nn_noniid13)

sys.stdout.close()



