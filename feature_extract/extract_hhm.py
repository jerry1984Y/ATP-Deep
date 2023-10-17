import os
import re
import numpy as np

def read_data_file_trip(filename):
    f = open(filename)
    data = f.readlines()
    f.close()

    results=[]
    block=len(data)//2
    for index in range(block):
        item1=data[index*2+0].split()
        name =item1[0].strip()
        seq=item1[1].strip()
        item2 = data[index * 2 + 1].split() #label
        item = []
        item.append(name)
        item.append(seq)
        results.append(item)
    return results

def extratdataa3m_hmm(file,desta3mfolder,desthmmfolder):
    student_tuples = read_data_file_trip(file)
    acid20 = 'ARNDCQEGHILKMFPSTWYV'
    for name, seq in student_tuples:
        with open(os.path.join(desta3mfolder, 'tmphhm.seq'), 'w') as f:
            f.write('>'+name+'\n'+seq)
            f.close()
        cmd = 'hhblits -i '
        cmd += os.path.join(desta3mfolder, 'tmphhm.seq')
        cmd+=' -oa3m '
        cmd+= os.path.join(desta3mfolder, name+'.a3m ')
        cmd += ' -ohhm '
        cmd += os.path.join(desthmmfolder, name + '.hhm')
        cmd+=' -n 3 -cpu 30 -d /home/dell/Documents/UniRef30_2022_02_hhsuite/UniRef30_2022_02'
        os.system(cmd)


if __name__ == '__main__':
    print('----prepare dataset------')

    extratdataa3m_hmm('../DataSet/atp-17-for-227.txt',
               '../DataSet/hhdataseta3m/',
                  '../DataSet/hhdatasethhm/')

    extratdataa3m_hmm('../DataSet/atp-41-for-388.txt',
                      '../DataSet/hhdataseta3m/',
                      '../DataSet/hhdatasethhm/')

    extratdataa3m_hmm('../DataSet/atp-227.txt',
                      '../DataSet/hhdataseta3m/',
                      '../DataSet/hhdatasethhm/')

    extratdataa3m_hmm('../DataSet/atp-388.txt',
                      '../DataSet/hhdataseta3m/',
                      '../DataSet/hhdatasethhm/')

    extratdataa3m_hmm('../DataSet/atp-549.txt',
                      '../DataSet/hhdataseta3m/',
                      '../DataSet/hhdatasethhm/')
    print('----finish-------')

