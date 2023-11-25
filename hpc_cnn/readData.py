results = {}
for i in range(1, 25):
    filename = f"data1124/G{i}.txt"
    f = open(filename, "r")
    lines = f.readlines()

    l = 0

    for line in lines:
        match l:
            case 0:
                oaccname = line.strip()
                results[oaccname] = []
            case 1:
                results[oaccname].append(line)
                
            case 2:
                accs_bins_name = line.strip()
                results[accs_bins_name] = []
            case 3:
                results[accs_bins_name].append(line)
                
            case 4:
                ppname = line.strip()
                results[ppname] = []
            case 5:
                results[ppname].append(line)
                
            case 6:
                ecename = line.strip()
                results[ecename] = []
            case 7:
                results[ecename].append(line)
                
            case 8:
                entroname = line.strip()
                results[entroname] = []
            case 9:
                results[entroname].append(line)
                
            case 10:
                entro_bins_name = line.strip()
                results[entro_bins_name] = []
            case 11:
                results[entro_bins_name].append(line)

            case 12:
                vr_name = line.strip()
                results[vr_name] = []
            case 13:
                results[vr_name].append(line)
                
        l += 1

# for key in results:
#     count_nan = results[key][0].count("nan")
#     print(f"Key: {key}, Count: {count_nan}")


for key in results:
    
    results[key] = results[key][0].replace("nan", "None") if "nan" in results[key][0] else results[key][0]
    results[key] = eval(results[key])
            

keys = list(results.keys())
values = list(results.values())
