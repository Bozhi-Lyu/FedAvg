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