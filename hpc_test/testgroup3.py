from FedAvg_2NN import *
import sys

filename = "G3.txt"
sys.stdout = open(filename, 'w')
mlp_iid3 = copy.deepcopy(mlp)
acc_mlp_iid3, predprob_2nn_iid3, ece_2nn_iid3, entropy_2nn_iid3, VR_2nn_iid3  = fedavg(mlp_iid3, C = 0.5, K = 100, E = 1, 
                      c_loader = iid_train_loader, rounds = 100, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_iid3")
print(acc_mlp_iid3)

print("predprob_2nn_iid3")
print(predprob_2nn_iid3)

print("ece_2nn_iid3")
print(ece_2nn_iid3)

print("entropy_2nn_iid3")
print(entropy_2nn_iid3)

print("VR_2nn_iid3")
print(VR_2nn_iid3)

sys.stdout.close()



filename = "G7.txt"
sys.stdout = open(filename, 'w')
mlp_noniid7 = copy.deepcopy(mlp)
acc_mlp_noniid7, predprob_2nn_noniid7, ece_2nn_noniid7, entropy_2nn_noniid7, VR_2nn_noniid7 = fedavg(mlp_noniid7, C = 0.5, K = 100, E = 1, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid7")
print(acc_mlp_noniid7)

print("predprob_2nn_noniid7")
print(predprob_2nn_noniid7)

print("ece_2nn_noniid7")
print(ece_2nn_noniid7)

print("entropy_2nn_noniid7")
print(entropy_2nn_noniid7)

print("VR_2nn_noniid7")
print(VR_2nn_noniid7)

sys.stdout.close()


filename = "G11.txt"
sys.stdout = open(filename, 'w')
mlp_noniid11 = copy.deepcopy(mlp)
acc_mlp_noniid11, predprob_2nn_noniid11, ece_2nn_noniid11, entropy_2nn_noniid11, VR_2nn_noniid11 = fedavg(mlp_noniid11, C = 0.5, K = 100, E = 10, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid11")
print(acc_mlp_noniid11)

print("predprob_2nn_noniid11")
print(predprob_2nn_noniid11)

print("ece_2nn_noniid11")
print(ece_2nn_noniid11)

print("entropy_2nn_noniid11")
print(entropy_2nn_noniid11)

print("VR_2nn_noniid11")
print(VR_2nn_noniid11)

sys.stdout.close()



filename = "G15.txt"
sys.stdout = open(filename, 'w')
mlp_noniid15 = copy.deepcopy(mlp)
acc_mlp_noniid15, predprob_2nn_noniid15, ece_2nn_noniid15, entropy_2nn_noniid15, VR_2nn_noniid15 = fedavg(mlp_noniid15, C = 0.5, K = 100, E = 20, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid15")
print(acc_mlp_noniid15)

print("predprob_2nn_noniid15")
print(predprob_2nn_noniid15)

print("ece_2nn_noniid15")
print(ece_2nn_noniid15)

print("entropy_2nn_noniid15")
print(entropy_2nn_noniid15)

print("VR_2nn_noniid15")
print(VR_2nn_noniid15)

sys.stdout.close()




