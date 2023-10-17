import torch
import esm
import numpy as np
import os


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
    student_tuples=read_data_file_trip(file)
    student_tuples=sorted(student_tuples, key=lambda student: student[2],reverse=True)
    # with open(os.path.join(destfolder,'file_41.txt'), 'w') as f:
    #     f.write('\n'.join(item[0] for item in student_tuples))

    for name,seq,_,label in student_tuples:
        # with open(os.path.join(destfolder,name+ '.label'), 'w') as f:
        #     f.write(','.join(l for l in label))
        print(name)
        data = [(name, seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            if torch.cuda.is_available():
                batch_tokens = batch_tokens.to(device="cuda:1")
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            if torch.cuda.is_available():
                results = results['representations'][33].to(device="cpu")
            else:
                results = results['representations'][33]

        token_representations = results

        for token_representation, tokens_len, batch_label in zip(token_representations, batch_lens, batch_labels):
            with open(os.path.join(destfolder, batch_label + '.data'), 'w') as f:
                np.savetxt(f, token_representation[1: tokens_len - 1], delimiter=',', fmt='%s')

if __name__ == "__main__":
    # Load ESM-2 model
    print('----loading esm-2 model-------')
    #esm2_t36_3B_UR50D esm2_t33_650M_UR50D
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    if torch.cuda.is_available():
        #model = model.cuda()
        model = model.to('cuda:0')
    print('----prepare dataset------')

    extratdata('../DataSet/atp-17-for-227.txt', '../DataSet/esm_embedding1280/')
    extratdata('../DataSet/atp-41-for-388.txt', '../DataSet/esm_embedding1280/')
    extratdata('../DataSet/atp-227.txt', '../DataSet/esm_embedding1280/')
    extratdata('../DataSet/atp-388.txt', '../DataSet/esm_embedding1280/')
    extratdata('../DataSet/atp-549.txt', '../DataSet/esm_embedding1280/')
    print('----finish-------')


