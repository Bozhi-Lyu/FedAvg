from FedAvg_2NN import *
import sys

filename = "G4.txt"
sys.stdout = open(filename, 'w')
mlp_iid4 = copy.deepcopy(mlp)
acc_mlp_iid4, predprob_2nn_iid4, ece_2nn_iid4, entropy_2nn_iid4, VR_2nn_iid4  = fedavg(mlp_iid4, C = 1, K = 100, E = 1, 
                      c_loader = iid_train_loader, rounds = 100, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_iid4")
print(acc_mlp_iid4)

print("predprob_2nn_iid4")
print(predprob_2nn_iid4)

print("ece_2nn_iid4")
print(ece_2nn_iid4)

print("entropy_2nn_iid4")
print(entropy_2nn_iid4)

print("VR_2nn_iid4")
print(VR_2nn_iid4)

sys.stdout.close()


filename = "G8.txt"
sys.stdout = open(filename, 'w')
mlp_noniid8 = copy.deepcopy(mlp)
acc_mlp_noniid8, predprob_2nn_noniid8, ece_2nn_noniid8, entropy_2nn_noniid8, VR_2nn_noniid8 = fedavg(mlp_noniid8, C = 1, K = 100, E = 1, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid8")
print(acc_mlp_noniid8)

print("predprob_2nn_noniid8")
print(predprob_2nn_noniid8)

print("ece_2nn_noniid8")
print(ece_2nn_noniid8)

print("entropy_2nn_noniid8")
print(entropy_2nn_noniid8)

print("VR_2nn_noniid8")
print(VR_2nn_noniid8)

sys.stdout.close()


filename = "G12.txt"
sys.stdout = open(filename, 'w')
mlp_noniid12 = copy.deepcopy(mlp)
acc_mlp_noniid12, predprob_2nn_noniid12, ece_2nn_noniid12, entropy_2nn_noniid12, VR_2nn_noniid12 = fedavg(mlp_noniid12, C = 1, K = 100, E = 10, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid12")
print(acc_mlp_noniid12)

print("predprob_2nn_noniid12")
print(predprob_2nn_noniid12)

print("ece_2nn_noniid12")
print(ece_2nn_noniid12)

print("entropy_2nn_noniid12")
print(entropy_2nn_noniid12)

print("VR_2nn_noniid12")
print(VR_2nn_noniid12)

sys.stdout.close()


filename = "G16.txt"
sys.stdout = open(filename, 'w')
mlp_noniid16 = copy.deepcopy(mlp)
acc_mlp_noniid16, predprob_2nn_noniid16, ece_2nn_noniid16, entropy_2nn_noniid16, VR_2nn_noniid16 = fedavg(mlp_noniid16, C = 0.1, K = 100, E = 20, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid16")
print(acc_mlp_noniid16)

print("predprob_2nn_noniid16")
print(predprob_2nn_noniid16)

print("ece_2nn_noniid16")
print(ece_2nn_noniid16)

print("entropy_2nn_noniid16")
print(entropy_2nn_noniid16)

print("VR_2nn_noniid16")
print(VR_2nn_noniid16)

sys.stdout.close()








