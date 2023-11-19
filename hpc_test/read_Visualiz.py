# To be modified. 2023.11.19
f = open("FedAvg_2NN_test_out.txt", "r")
lines = f.readlines()

l = 0
results = {}
for line in lines:
    match (l % 7):
        case 0:
            accname = line.strip()
            results[accname] = []
        case 1:
            results[accname].append(line)
        case 2:
            entropyname = line.strip()
            results[entropyname] = []
        case 3:
            results[entropyname].append(line)
        case 4:
            VRname = line.strip()
            results[VRname] = []
        case 5:
            results[VRname].append(line)

    l += 1

for key in results:
    results[key] = eval(results[key][0])
