import pandas as pd
res = pd.DataFrame()

mean_speed_list = []
Mean_Time_Per_Step_list = []
speed_uncertainty_list = []
Speed_Jitter_list = []
gpu_type_list = []
num_gpu_list = []
staged_from_list = []

import sys
import os
import re

cwd = os.getcwd()

rootdir = cwd + '/results/'
count = -1
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if 'archived' in subdir:
            continue
        count += 1
        current_file = os.path.join(subdir, file)
        print("Reading Output File : " + current_file)
        file_object  = open(current_file, 'r')
        for i in file_object.readlines():
            if 'Mean Speed' in i:
                current_line = i 
                mean_speed = re.findall(r"[-+]?\d*\.\d+|\d+", current_line)
                mean_speed_list.append(float(mean_speed[0]))
            if 'Mean Time Per Step' in i:
                current_line = i
                Mean_Time_Per_Step = re.findall(r"[-+]?\d*\.\d+|\d+", current_line)
                Mean_Time_Per_Step_list.append(float(Mean_Time_Per_Step[0]))
            if 'speed uncertainty' in i:
                current_line = i
                speed_uncertainty = re.findall(r"[-+]?\d*\.\d+|\d+", current_line)
                speed_uncertainty_list.append(float(speed_uncertainty[0]))
            if 'Speed Jitter' in i:
                current_line = i
                Speed_Jitter = re.findall(r"[-+]?\d*\.\d+|\d+", current_line)
                Speed_Jitter_list.append(float(Speed_Jitter[0]))
            if 'Gres=' in i:
                current_line = i
                gpu_type = re.search('gpu:(.*):', current_line).group(1)
                num_gpu = re.search(gpu_type + ':(.*) ', current_line).group(1)
                gpu_type_list.append(gpu_type)
                num_gpu_list.append(num_gpu)
            if 'Command=' in i:
                current_line = i
                staged_from = '$SCRATCH'
                if 'local' in current_line:
                    staged_from = '$LOCAL'
                staged_from_list.append(staged_from)

res['mean_speed'] = mean_speed_list
res['Mean_Time_Per_Step'] = Mean_Time_Per_Step_list
res['speed_uncertainty'] = speed_uncertainty_list
res['Speed_Jitter'] = Speed_Jitter_list
res['gpu_type'] = gpu_type_list
res['num_gpu'] = num_gpu_list
res['staged_from'] = staged_from_list

print('Creating CVS File Output')
import datetime
now = str(datetime.date.today())
outfilename = rootdir + now +':' + 'TF_PERFORMANCE_RESULTS.csv'
res.to_csv(path_or_buf=outfilename,sep=',')
print('CSV File Created At : ' + outfilename)

#Creating Graphs

res_k80 = res.loc[res['gpu_type'] == 'k80']
res_p100 = res.loc[res['gpu_type'] == 'p100']
res_k80_local = res_k80.loc[res['staged_from']=='$LOCAL']
res_k80_scratch = res_k80.loc[res['staged_from']=='$SCRATCH']
res_p100_local = res_p100.loc[res['staged_from']=='$LOCAL']
res_p100_scratch = res_p100.loc[res['staged_from']=='$SCRATCH']

#For k80 and SCRATCH
y = list(res_k80_scratch['mean_speed'])
x = list(res_k80_scratch['num_gpu'])
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.bar(x,y)
plt.xlabel('Num_GPUs')
plt.ylabel('Mean Speed : Images/Sec')
plt.title('GPU TYPE : k80 , Directory : $SCRATCH')
for a,b in zip(x, y):
    plt.text(a, b, str("{:.2f}".format(b)))
plt.rcParams["figure.figsize"] = (8,8)
filepath_image = rootdir + now + ':' + 'K80-SCRATCH.png'
plt.savefig(filepath_image)

#For K80 and Local
y = list(res_k80_local['mean_speed'])
x = list(res_k80_local['num_gpu'])
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.bar(x,y)
plt.xlabel('Num_GPUs')
plt.ylabel('Mean Speed : Images/Sec')
plt.title('GPU TYPE : k80 , Directory : $LOCAL')
for a,b in zip(x, y):
    plt.text(a, b, str("{:.2f}".format(b)))
plt.rcParams["figure.figsize"] = (8,8)
filepath_image = rootdir + now + ':' + 'K80-LOCAL.png'
plt.savefig(filepath_image)

#For p100 and LOCAL
y = list(res_p100_local['mean_speed'])
x = list(res_p100_local['num_gpu'])
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.bar(x,y)
plt.xlabel('Num_GPUs')
plt.ylabel('Mean Speed : Images/Sec')
plt.title('GPU TYPE : p100 , Directory : $LOCAL')
for a,b in zip(x, y):
    plt.text(a, b, str("{:.2f}".format(b)))
plt.rcParams["figure.figsize"] = (8,8)
filepath_image = rootdir + now + ':' + 'P100-LOCAL.png'
plt.savefig(filepath_image)

#For p100 and SCRATCH
y = list(res_p100_scratch['mean_speed'])
x = list(res_p100_scratch['num_gpu'])
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.bar(x,y)
plt.xlabel('Num_GPUs')
plt.ylabel('Mean Speed : Images/Sec')
plt.title('GPU TYPE : p100 , Directory : $SCRATCH')
for a,b in zip(x, y):
    plt.text(a, b, str("{:.2f}".format(b)))
plt.rcParams["figure.figsize"] = (8,8)
filepath_image = rootdir + now + ':' + 'P100-SCRATCH.png'
plt.savefig(filepath_image)

print("Graphs Created At : " + rootdir)
#End of code
