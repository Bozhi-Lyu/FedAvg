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