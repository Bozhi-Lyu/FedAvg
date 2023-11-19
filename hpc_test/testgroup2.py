from FedAvg_2NN import *
import sys

filename = "G2.txt"
sys.stdout = open(filename, 'w')
mlp_iid2 = copy.deepcopy(mlp)
acc_mlp_iid2, predprob_2nn_iid2, ece_2nn_iid2, entropy_2nn_iid2, VR_2nn_iid2  = fedavg(mlp_iid2, C = 0.2, K = 100, E = 1, 
                      c_loader = iid_train_loader, rounds = 100, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_iid2")
print(acc_mlp_iid2)

print("predprob_2nn_iid2")
print(predprob_2nn_iid2)

print("ece_2nn_iid2")
print(ece_2nn_iid2)

print("entropy_2nn_iid2")
print(entropy_2nn_iid2)

print("VR_2nn_iid2")
print(VR_2nn_iid2)

sys.stdout.close()


filename = "G6.txt"
sys.stdout = open(filename, 'w')
mlp_noniid6 = copy.deepcopy(mlp)
acc_mlp_noniid6, predprob_2nn_noniid6, ece_2nn_noniid6, entropy_2nn_noniid6, VR_2nn_noniid6 = fedavg(mlp_noniid6, C = 0.2, K = 100, E = 1, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid6")
print(acc_mlp_noniid6)

print("predprob_2nn_noniid6")
print(predprob_2nn_noniid6)

print("ece_2nn_noniid6")
print(ece_2nn_noniid6)

print("entropy_2nn_noniid6")
print(entropy_2nn_noniid6)

print("VR_2nn_noniid6")
print(VR_2nn_noniid6)

sys.stdout.close()



filename = "G10.txt"
sys.stdout = open(filename, 'w')
mlp_noniid10 = copy.deepcopy(mlp)
acc_mlp_noniid10, predprob_2nn_noniid10, ece_2nn_noniid10, entropy_2nn_noniid10, VR_2nn_noniid10 = fedavg(mlp_noniid10, C = 0.2, K = 100, E = 10, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid10")
print(acc_mlp_noniid10)

print("predprob_2nn_noniid10")
print(predprob_2nn_noniid10)

print("ece_2nn_noniid10")
print(ece_2nn_noniid10)

print("entropy_2nn_noniid10")
print(entropy_2nn_noniid10)

print("VR_2nn_noniid10")
print(VR_2nn_noniid10)

sys.stdout.close()


filename = "G14.txt"
sys.stdout = open(filename, 'w')
mlp_noniid14 = copy.deepcopy(mlp)
acc_mlp_noniid14, predprob_2nn_noniid14, ece_2nn_noniid14, entropy_2nn_noniid14, VR_2nn_noniid14 = fedavg(mlp_noniid14, C = 0.2, K = 100, E = 20, 
                      c_loader = noniid_train_loader, rounds = 600, 
                      lr = 0.05, acc_threshold = acc_threshold_2nn)

print("acc_mlp_noniid14")
print(acc_mlp_noniid14)

print("predprob_2nn_noniid14")
print(predprob_2nn_noniid14)

print("ece_2nn_noniid14")
print(ece_2nn_noniid14)

print("entropy_2nn_noniid14")
print(entropy_2nn_noniid14)

print("VR_2nn_noniid14")
print(VR_2nn_noniid14)

sys.stdout.close()








