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