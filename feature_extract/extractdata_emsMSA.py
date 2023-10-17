import os
import string
from typing import Tuple, List
from Bio import SeqIO
import glob
import esm
from scipy.spatial.distance import squareform, pdist, cdist
import torch
import numpy as np
import math

deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str) -> List[Tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]

f = glob.glob(r'../DataSet/hhdataseta3m/*.a3m')
PDB_IDS=[]
for item in f:
    fname=os.path.splitext(os.path.split(item)[-1])[0]
    #print(fname)
    PDB_IDS.append(fname)




msas = {
    name: read_msa(f"../DataSet/hhdataseta3m/{name}.a3m")
    for name in PDB_IDS
}

# sequences = {
#     name: msa[0] for name, msa in msas.items()
# }


# Select sequences from the MSA to maximize the hamming distance
# Alternatively, can use hhfilter
def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa

    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]

msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_transformer=msa_transformer.eval().cuda()
msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()

msa_transformer_predictions = {}
msa_transformer_results = []
for name, inputs in msas.items():
    if os.path.exists(os.path.join('../DataSet/msa_embedding/', name + '.data')):
        print(name +' exist,pass')
        continue
    inputs = greedy_select(inputs, num_seqs=128) # can change this to pass more/fewer sequences
    #print(inputs)
    #check content legnth, if >1024 split
    name, content = inputs[0]
    clen=len(content)
    maxlength=1023
    if clen>maxlength:
        segment=math.ceil(clen/maxlength)
        re=None
        for i in range(segment):
            if i==segment-1:
                inp=[(row0, row1[i*maxlength:]) for (row0, row1) in inputs]
            else:
                inp = [(row0, row1[i*maxlength:(i+1)*maxlength]) for (row0, row1) in inputs]
            msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = msa_transformer_batch_converter(
                [inp])
            msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).device)
            with torch.no_grad():
                results = msa_transformer(msa_transformer_batch_tokens, repr_layers=[12])
                token_representations = results["representations"]
                print(name+ ' segment')
                if i==0:
                    re=token_representations[12][0, 0, 1:, :].cpu()
                    print(re.shape)
                else:
                    re=torch.cat((re,token_representations[12][0, 0, 1:, :].cpu()),dim=0)
                    print(re.shape)

        with open(os.path.join('../DataSet/msa_embedding/', name + '.data'), 'w') as f:
            np.savetxt(f, re.numpy(), delimiter=',', fmt='%s')
    else:
        msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = msa_transformer_batch_converter([inputs])
        msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).device)
        with torch.no_grad():
            results=msa_transformer(msa_transformer_batch_tokens,repr_layers=[12])
            token_representations = results["representations"]
            print(name)
            with open(os.path.join('../DataSet/msa_embedding/', name + '.data'), 'w') as f:
                 np.savetxt(f, token_representations[12][0, 0, 1:, :].cpu().numpy(), delimiter=',', fmt='%s')

