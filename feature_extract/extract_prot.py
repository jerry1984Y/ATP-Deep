from time import sleep

import torch
from transformers import T5EncoderModel, T5Tokenizer

import re
import numpy as np
import gc
import os
import pandas as pd

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
        item2 = data[index * 2 + 1].split()
        item = []
        item.append(name)
        item.append(seq)
        item.append(len(seq))
        item.append(item2[1].strip())
        results.append(item)
    return results
def extratdata(file,destfolder):
    student_tuples = read_data_file_trip(file)
    student_tuples = sorted(student_tuples, key=lambda student: student[2], reverse=True)
    # with open(os.path.join(destfolder, 'file_prot.txt'), 'w') as f:
    #     f.write('\n'.join(item[0] for item in student_tuples))
    i=1
    for name, seq, length, label in student_tuples:
        print(i)
        i+=1

        with open(os.path.join(destfolder, name + '.label'), 'w') as f:
            f.write(','.join(l for l in label))
        newseq=' '.join(s for s in seq)
        newseq=re.sub(r"[UZOB]", "X", newseq)
        #print('newseq length',len(newseq))
        ids = tokenizer.batch_encode_plus([newseq], add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)

        embedding = embedding.last_hidden_state.cpu().numpy()
        features = []
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            #features.append(seq_emd)
            with open(os.path.join(destfolder, name + '.data'), 'w') as f:
                np.savetxt(f, seq_emd, delimiter=',', fmt='%s')

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        # sleep(0.5)
        # torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
        #model.to('cpu')

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cuda:0'
    model = model.to(device)
    model = model.eval()
    print('----prepare dataset------')

    extratdata('../DataSet/atp-17-for-227.txt', '../DataSet/prot_embedding/')
    extratdata('../DataSet/atp-41-for-388.txt', '../DataSet/prot_embedding/')
    extratdata('../DataSet/atp-227.txt', '../DataSet/prot_embedding/')
    extratdata('../DataSet/atp-388.txt', '../DataSet/prot_embedding/')
    extratdata('../DataSet/atp-549.txt', '../DataSet/prot_embedding/')
    print('----finish-------')
