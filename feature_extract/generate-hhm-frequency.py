# fetch name list
import glob
import os

hhm_path = '../DataSet/hhdatasethhm/'
hhm_path_files = os.listdir(hhm_path)
#f = glob.iglob(r'DataSet/hhdatasethhm/*.hhm')
name_list = []
for fi in hhm_path_files:
    hhm_name,ext = fi.split('.')
    if ext=='hhm':
        name_list.append(hhm_name)
print(len(name_list))

# generate mean space-hhblits
import numpy as np
import math

max_range = 0
output_path = '../DataSet/hhdatasethhm/'

for uniprot_id in name_list:
    # fetch length
    with open(hhm_path + uniprot_id + '.hhm') as hhm_file:
        hhm_line = hhm_file.readline()
        while hhm_line:
            if (hhm_line[0:4] == 'LENG'):
                hhm_seq_len = int(hhm_line.split()[1])
                break
            hhm_line = hhm_file.readline()
    # fetch 30d feature from .hhm
    with open(hhm_path + uniprot_id + '.hhm') as hhm_file:
        hhm_matrix = np.zeros([hhm_seq_len, 30], float)
        hhm_line = hhm_file.readline()
        idxx = 0
        while (hhm_line[0] != '#'):
            hhm_line = hhm_file.readline()
        for i in range(0, 5):
            hhm_line = hhm_file.readline()
        while hhm_line:
            if (len(hhm_line.split()) == 23):
                idxx += 1
                if (idxx == hhm_seq_len + 1):
                    break
                each_item = hhm_line.split()[2:22]
                for idx, s in enumerate(each_item):
                    if (s == '*'):
                        each_item[idx] = '99999'
                for j in range(0, 20):
                    try:
                        hhm_matrix[idxx - 1, j] = math.pow(2, float(int(each_item[j])) / (-1000))
                    except IndexError:
                        pass
            elif (len(hhm_line.split()) == 10):
                each_item = hhm_line.split()[0:10]
                for idx, s in enumerate(each_item):
                    if (s == '*'):
                        each_item[idx] = '99999'
                for j in range(20, 30):
                    try:
                        hhm_matrix[idxx - 1, j] = math.pow(2, float(int(each_item[j - 20])) / (-1000))
                    except IndexError:
                        pass
            hhm_line = hhm_file.readline()
        with open(output_path + uniprot_id + '.shhfeature', 'w') as out_file:
            np.savetxt(out_file, hhm_matrix, fmt='%.6f', delimiter=',')
