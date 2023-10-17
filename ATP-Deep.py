import math

import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import torch
import torch.nn as nn

from sklearn import metrics

from LossFunction.focalLoss import FocalLoss_v2
from utility import read_file_list, save_prob_label, masked_softmax, read_file_list_from_seq_label


class Contextual_Attention(nn.Module):
    def __init__(self ,q_input_dim,v_input_dim=1024,qk_dim=96,v_dim=96):
        super(Contextual_Attention,self).__init__()
        self.cn3=nn.Conv1d(q_input_dim,qk_dim,3, padding='same')
        self.cn5 = nn.Conv1d(q_input_dim, qk_dim, 5, padding='same')
        self.k = nn.Linear(v_dim * 2+q_input_dim, qk_dim)
        self.q=nn.Linear(q_input_dim,qk_dim)
        self.v=nn.Linear(v_input_dim,v_dim)
        self._norm_fact = 1 / torch.sqrt(torch.tensor(qk_dim))

    def forward(self, plm_embedding,evo_local,seqlengths):
        Q = self.q(evo_local)#self.q(evo_local)  # Q: batch_size * seq_len * dim_k
        k3=self.cn3(evo_local.permute(0,2,1))
        k5 = self.cn5(evo_local.permute(0, 2, 1))
        evo_local=torch.cat((evo_local,k3.permute(0,2,1),k5.permute(0,2,1)),dim=2)
        K = self.k(evo_local)  # K: batch_size * seq_len * qk_dim
        V = self.v(plm_embedding)  # V: batch_size * seq_len * dim_v

        # atten = nn.Softmax(dim=-1)(torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact  # Q * K.T() # batch_size * seq_len * seq_len
        atten=masked_softmax((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact,seqlengths)
        output = torch.bmm(atten, V)  # Q * K.T() * V # batch_size * seq_len * dim_v

        return output+V


def coll_paddding(batch_traindata):
    batch_traindata.sort(key=lambda data: len(data[0]), reverse=True)
    feature0 = []
    f0agv=[]
    feature_hmm = []

    train_y = []

    for data in batch_traindata:
        feature0.append(data[0])
        f0agv.append(data[1])
        feature_hmm.append(data[2])
        train_y.append(data[3])
    data_length = [len(data) for data in feature0]

    mask = torch.full((len(batch_traindata), data_length[0]), False).bool()  # crete init mask
    for mi, aci in zip(mask, data_length):
        mi[aci:] = True

    feature0 = torch.nn.utils.rnn.pad_sequence(feature0, batch_first=True, padding_value=0)
    f0agv = torch.nn.utils.rnn.pad_sequence(f0agv, batch_first=True, padding_value=0)
    feature_hmm = torch.nn.utils.rnn.pad_sequence(feature_hmm, batch_first=True, padding_value=0)
    train_y = torch.nn.utils.rnn.pad_sequence(train_y, batch_first=True, padding_value=0)
    return feature0,f0agv,feature_hmm,train_y,torch.tensor(data_length)


class BioinformaticsDataset(Dataset):
    # X: list of filename
    def __init__(self, X):
        self.X = X
    def __getitem__(self, index):
        filename = self.X[index]
        #esm_embedding1280 prot_embedding  esm_embedding2560 msa_embedding
        df0 = pd.read_csv('DataSet/prot_embedding/' + filename + '.data', header=None)
        prot = df0.values.astype(float).tolist()

        prot = torch.tensor(prot)
        agv = torch.mean(prot, dim=0)
        # print(agv)
        agv = agv.repeat(prot.shape[0], 1)


        #hhdatasethhm pssm20 hhdataseta3m
        dfhmm = pd.read_csv('DataSet/hhdatasethhm' + '/' + filename + '.shhfeature', header=None)
        feature_hmm = dfhmm.values.astype(float).tolist()

        df2= pd.read_csv('DataSet/prot_embedding/'+  filename+'.label', header=None)
        label = df2.values.astype(int).tolist()
        label = torch.tensor(label)
        #reduce 2D to 1D
        label=torch.squeeze(label)

        return prot,agv,torch.tensor(feature_hmm), label


    def __len__(self):
        return len(self.X)

class DCTModule(nn.Module):
    def __init__(self):
        super(DCTModule, self).__init__()
        #20-256-512-256-128-l512-l64-l2  3,5,3,5
        # 21 psfm
        # 30 hmm
        # 20 pssm
        # 13 chem
        self.ca=Contextual_Attention(q_input_dim=30,v_input_dim=1024)

        self.relu=nn.ReLU(True)
        #1024+32+30+13,512 96
        self.protcnn1=nn.Conv1d(1024+1024+96,512,3,padding='same')
        self.protcnn2 = nn.Conv1d(512, 256, 3, padding='same')
        self.protcnn3 = nn.Conv1d(256, 128, 3, padding='same')


        self.fc2= nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, 2)
        self.drop=nn.Dropout(0.5)


    def forward(self,prot0,f0agv,evo,data_length):
        evosa=self.ca(prot0,evo,data_length)

        prot=torch.cat((prot0,f0agv,evosa),dim=2)

        prot = prot.permute(0, 2, 1)
        prot=self.protcnn1(prot)
        prot=self.relu(prot)
        prot=self.protcnn2(prot)
        prot = self.relu(prot)
        prot = self.protcnn3(prot)
        prot = self.relu(prot)
        prot = prot.permute(0, 2, 1)
        x=self.fc2(prot)
        x=self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.fc4(x)
        return x


def train():
    train_set = BioinformaticsDataset(train_file_list)
    model = DCTModule()
    epochs =20

    model = model.to(device)
    train_loader = DataLoader(dataset=train_set, batch_size=16,shuffle=True, num_workers=16,  pin_memory=True, persistent_workers=True,
                              collate_fn=coll_paddding)
    best_val_loss = 3000

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    per_cls_weights = torch.FloatTensor([0.2,0.8]).to(device)
    fcloss=FocalLoss_v2(alpha=per_cls_weights, gamma=2)
    model.train()
    for i in range(epochs):

        epoch_loss_train = 0.0
        nb_train = 0
        for prot_x,f0agv,evo_x,data_y,length in train_loader:
            optimizer.zero_grad()
            y_pred = model(prot_x.to(device),f0agv.to(device),evo_x.to(device),length.to(device))
            y_pred = torch.nn.utils.rnn.pack_padded_sequence(y_pred, length.to('cpu'), batch_first=True)
            data_y = torch.nn.utils.rnn.pack_padded_sequence(data_y, length, batch_first=True)
            data_y = data_y.to(device)

            single_loss=fcloss(y_pred.data,data_y.data)

            single_loss.backward()
            optimizer.step()
            epoch_loss_train = epoch_loss_train + single_loss.item()
            nb_train+=1
        epoch_loss_avg=epoch_loss_train/nb_train
        if best_val_loss > epoch_loss_avg:
            model_fn = "protein_Deep_ATP.pkl"
            torch.save(model.state_dict(), model_fn)
            best_val_loss = epoch_loss_avg
            if i % 10 == 0:
                print('epochs ', i)
                print("Save model, best_val_loss: ", best_val_loss)
def test():
    test_set = BioinformaticsDataset(test_file_list)

    test_load = DataLoader(dataset=test_set, batch_size=16,
                           num_workers=16, pin_memory=True, persistent_workers=True, collate_fn=coll_paddding)
    model = DCTModule()
    model = model.to(device)

    print("==========================Test RESULT================================")

    model.load_state_dict(torch.load('protein_Deep_ATP.pkl'))
    model.eval()

    arr_probs = []
    arr_labels = []
    arr_labels_hyps = []
    with torch.no_grad():
        for prot_x,f0agv,evo_x,data_y, length in test_load:
            y_pred = model(prot_x.to(device),f0agv.to(device),evo_x.to(device),length.to(device))
            y_pred = torch.nn.utils.rnn.pack_padded_sequence(y_pred, length.to('cpu'), batch_first=True)
            y_pred=y_pred.data
            y_pred=torch.nn.functional.softmax(y_pred,dim=1)
            arr_probs.extend(y_pred[:, 1].to('cpu'))
            y_pred=torch.argmax(y_pred, dim=1).to('cpu')
            data_y = torch.nn.utils.rnn.pack_padded_sequence(data_y, length, batch_first=True)
            arr_labels.extend(data_y.data)
            arr_labels_hyps.extend(y_pred)

    print('-------------->')

    auc =metrics.roc_auc_score(arr_labels, arr_probs)
    print('acc ', metrics.accuracy_score(arr_labels, arr_labels_hyps))
    print('balanced_accuracy ', metrics.balanced_accuracy_score(arr_labels, arr_labels_hyps))
    tn, fp, fn, tp = metrics.confusion_matrix(arr_labels, arr_labels_hyps).ravel()
    print('tn, fp, fn, tp ',tn, fp, fn, tp )
    print('MCC ', metrics.matthews_corrcoef(arr_labels, arr_labels_hyps))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1score = 2 * tp / (2 * tp + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    youden = sensitivity + specificity - 1
    print('sensitivity ', sensitivity)
    print('specificity ', specificity)
    print('precision ', precision)
    print('recall ', recall)
    print('f1score ', f1score)
    print('youden ', youden)
    print('auc', auc)
    print('<----------------')
    save_prob_label(arr_probs, arr_labels, 'protT5_avg_cxt_hmm_549_41.csv')
    print('<----------------save to csv finish')


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    torch.cuda.set_device(0)
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")

    train_file_list = read_file_list_from_seq_label('DataSet/atp-549.txt')
    test_file_list = read_file_list_from_seq_label('DataSet/atp-41-for-388.txt')
    train()
    test()

