import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from tqdm import tqdm
import ipdb
from torch import tanh,sigmoid
from torch.nn.functional import leaky_relu
from torch.utils.tensorboard import SummaryWriter
import os
# import pickle as pkl

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------- #
#                           Data Import & Processing                           #
# ---------------------------------------------------------------------------- #

# ---------------------------------- Import ---------------------------------- #
data_temp=pd.read_csv("tempAMAL_train.csv")

# ------------------------------- Preprocessing ------------------------------ #

data_temp=data_temp.iloc[:,1:]
data_temp.fillna(method="ffill",inplace=True)
cities=data_temp.columns
data_temp=np.array(data_temp)

# ------------------------------ MinMax scaling ------------------------------ #

data_temp=(data_temp-np.min(data_temp))/(np.max(data_temp)-np.min(data_temp))

# ---------------------------------------------------------------------------- #
#                               Class definition                               #
# ---------------------------------------------------------------------------- #

# -------------------- Recurrent Neural Network definition ------------------- #
class NN(nn.Module):
    def __init__(self,inSize,outSize,layers=[]):
        super(NN,self).__init__()
        self.layers=nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize,x))
            inSize=x
        self.layers.append(nn.Linear(inSize,outSize))

    def forward(self,x):
        x=self.layers[0](x)
        for i in range(1,len(self.layers)):
            x=leaky_relu(x)
            x=self.layers[i](x)
        return x            

inSize=2
outSize=1
layers=[20,20,10]
x=torch.randn(size=(10,2))
test=NN(inSize,outSize,layers=layers)
print(test.forward(x))


class RNN(torch.nn.Module):
    
    def __init__(self,dim,latent,output):
        super().__init__()
        self.dim=dim
        self.latent=latent
        self.output=output
        self.lin_i=torch.nn.Linear(dim,latent)
        self.lin_h=torch.nn.Linear(latent,latent,bias=False)
        self.lin_d=torch.nn.Linear(latent,output)

    def one_step(self,x,h):
        #computes h 
        return tanh(self.lin_i(x)+self.lin_h(h))
    
    def forward(self,x,h):
        sequence_length=x.shape[0]
        for i in range(sequence_length):
            # ipdb.set_trace()
            h=self.one_step(x[i],h)
        return sigmoid(self.lin_d(h))
    
    # def decode(self,h):
    #     return sigmoid(self.lin_d(h))
    
# ---------------------- Sequence Classifier definition ---------------------- #

class SequenceClassifier():

    def __init__(self,dim,latent,output):
        self.dim,self.latent,self.output=dim,latent,output
        self.rnn=RNN(dim,latent,output)
        # self.model=
    
    def forward(self,x):
        h=torch.zeros(size=(self.latent))
        x=x.view(x.shape[0],x.shape[1],self.dim)
        y=self.rnn.forward(x,h)
        return x

# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #

if __name__=="__main__":
    sequence_length=100
    train_batch=100
    test_batch=200




