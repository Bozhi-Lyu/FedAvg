import os

file_count = 9
file_prefix = 'G'


template1 = '''\
from FedAvg_CNN import *
import sys

# overall_acc: oacc_i{file_number}
# accs_bins_of_rounds: accs_bins_i{file_number}
# predprob_of_rounds: pp_i{file_number}
# ece_of_rounds: ece_i{file_number}
# entropys_of_rounds: entro_i{file_number}
# entrophys_bins_of_rounds: entro_bins_i{file_number}
# VR_of_rounds: VR_i{file_number}
# oe_of_rounds: oe_i{file_number}
# counts_of_rounds: counts_i{file_number}

filename = "G{file_number}.txt"
sys.stdout = open(filename, 'w')

GroupNum = [1, 2, 3, 4, 5, 6, 7, 8, 9]
C = [0.1, 0.5, 1, 0.1, 0.5, 1, 0.1, 0.5, 1]
E = [1, 1, 1, 10, 10, 10, 20, 20, 20]

cnn_iid{file_number} = copy.deepcopy(cnn)
a1, a2, a3, a4, a5, a6, a7, a8, a9 = fedavg(cnn_iid{file_number}, 
                                    C = C[{file_number} - 1], 
                                    K = 100, 
                                    E = E[{file_number} - 1], 
                                    c_loader = iid_train_loader, 
                                    rounds = 1200, 
                                    lr = 0.05, 
                                    acc_threshold = acc_threshold_cnn)

print("oacc_i{file_number}")
print(a1)

print("accs_bins_i{file_number}")
print(a2)

print("pp_i{file_number}")
print(a3)

print("ece_i{file_number}")
print(a4)

print("entro_i{file_number}")
print(a5)

print("entro_bins_i{file_number}")
print(a6)

print("VR_i{file_number}")
print(a7)

print("oe_i{file_number}")
print(a8)

print("counts_i{file_number}")
print(a9)

sys.stdout.close()
'''


template2 = '''\
from FedAvg_CNN import *
import sys

# overall_acc: oacc_i{file_number}
# accs_bins_of_rounds: accs_bins_i{file_number}
# predprob_of_rounds: pp_i{file_number}
# ece_of_rounds: ece_i{file_number}
# entropys_of_rounds: entro_i{file_number}
# entrophys_bins_of_rounds: entro_bins_i{file_number}
# VR_of_rounds: VR_i{file_number}
# oe_of_rounds: oe_i{file_number}
# counts_of_rounds: counts_i{file_number}

filename = "G{file_number}.txt"
sys.stdout = open(filename, 'w')

GroupNum = [10, 11, 12, 13, 14, 15, 16, 17, 18]
C = [0.1, 0.5, 1, 0.1, 0.5, 1, 0.1, 0.5, 1]
E = [1, 1, 1, 10, 10, 10, 20, 20, 20]

cnn_noniid{file_number} = copy.deepcopy(cnn)
a1, a2, a3, a4, a5, a6, a7, a8, a9  = fedavg(cnn_noniid{file_number}, 
                                    C = C[{file_number} - {file_count} - 1], 
                                    K = 100, 
                                    E = E[{file_number} - {file_count} - 1], 
                                    c_loader = noniid_train_loader, 
                                    rounds = 2000, 
                                    lr = 0.05, 
                                    acc_threshold = acc_threshold_cnn)

print("oacc_n{file_number}")
print(a1)

print("accs_bins_n{file_number}")
print(a2)

print("pp_n{file_number}")
print(a3)

print("ece_n{file_number}")
print(a4)

print("entro_n{file_number}")
print(a5)

print("entro_bins_n{file_number}")
print(a6)

print("VR_n{file_number}")
print(a7)

print("oe_n{file_number}")
print(a8)

print("counts_n{file_number}")
print(a9)

sys.stdout.close()
'''

for i in range(1, file_count + 1):
    file_name = f"{file_prefix}{i}.py"
    file_path = os.path.join(os.getcwd(), file_name)
    
    file_content = template1.format(file_number = i, file_count = file_count)
    with open(file_path, 'w') as file:
        file.write(file_content)
        
        
        
for i in range(file_count + 1, 2 * file_count + 1):
    file_name = f"{file_prefix}{i}.py"
    file_path = os.path.join(os.getcwd(), file_name)
    
    file_content = template2.format(file_number = i, file_count = file_count)
    with open(file_path, 'w') as file:
        file.write(file_content)

