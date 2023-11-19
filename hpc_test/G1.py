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