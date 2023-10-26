# ATP-Deep
Prediction of Protein-ATP Binding Residues Using Multi-view Feature Learning via Contextual-based Co-attention Network

# 1. Citation
coming soon

# 2. Requirements

Python >= 3.10.6

torch = 2.0.0

pandas = 2.0.0

scikit-learn = 1.2.2

HHblits = 3.3.0

ProtTrans (ProtT5-XL-UniRef50 model)


# 3. Description
The proposed ATP-Deep method is implemented using python on torch for 
Protein-ATP binding residue prediction. 
The ATP-Deep model utilizes a multi-view features learning strategy to fuse 
the protein language models and domain-specific evolutionary context embedding via 
a contextual-based co-attention network and augmented residue 
feature with its corresponding protein-level information.

# 4 Datasets

atp-388.txt: this file contains 388 ATP binding proteins used for model training

atp-41-for-388.txt: this file contains 41 ATP binding proteins used for model testing

atp-227.txt:this file contains 227 ATP binding proteins used for model training

atp-17-for-227.txt: this file contains 17 ATP binding proteins used for model testing

atp-549.txt: this file contains 549 ATP binding proteins used for model training


# 5. How to Use

## 5.1 Set up environment for HMM and ProtTrans
1. Download hh-suite v3.3.0 from https://github.com/soedinglab/hh-suite and compile the source.
2. Download UniRef30_2022_02 dataset from https://gwdu111.gwdg.de/~compbiol/uniclust/2023_02/ .
3. Set ProtTrans follow procedure from https://github.com/agemagician/ProtTrans/tree/master.

## 5.2 Extract features

1. Extract HMM feature: cd to the ATP-Deep/feature_extract dictionary, 
and run "python3 extract_hhm.py and python3 generate-hhm-frequency.py",
the HMM matrixs will be extracted to Dataset/hhdatasethhm fold.

2. Extract pLMs embedding: cd to the ATP-Deep/feature_extract dictionary, 
and run "python3 extract_prot.py", the pLMs embedding matrixs will be extracted to Dataset/prot_embedding fold.

## 5.3 Train and test

1. cd to the ATP-Deep dictionary,and run "python3 ATP-Deep.py" for training and testing the model.

