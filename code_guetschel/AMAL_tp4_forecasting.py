import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from rnn import RNN
from torch_utils import  NN, validation_regress
from AMAL_tp4_datasets import DataHolder


##############################################################################
#### Networks
##############################################################################

## entrées sous forme length × batch × dim

class SequenceForecaster(nn.Module):
    def __init__(self, inDim, hidenDim, fc_layers, outDim):
        super().__init__()
        self.inDim, self.hidenDim, self.fc_layers , self.outDim = inDim, hidenDim, fc_layers, outDim

        self.rnn = RNN(inDim, hidenDim)
        self.fc = NN(hidenDim, outDim, fc_layers)

    def forward(self, x):      ## x : batch*seq_len*inDim
        h = - torch.ones(self.hidenDim) ## valeur unique (donées normalisées entre 0 et 1)
        x = x.view(x.shape[0], x.shape[1], self.inDim)
        x.transpose_(0,1)  ## now x : seq_len*batch*inDim
        x = self.rnn(x, h, many_to_many=True)
        x = self.fc(x)
        x.transpose_(0,1)  ## now x : batch*seq_len*inDim
        return x



##############################################################################
#### main
##############################################################################

if __name__=="__main__":
    ## imports :
    from torch.utils.tensorboard import SummaryWriter
    import os
    import pickle as pkl

    ## load dataset temperatures :
    path="./tempAMAL_train.csv"
    max_sample_length = 500
    train_batch = 100
    test_batch  = 100

    max_num_city = 2
    one_city = False ## tels if temp curves are considered one by one or not

    data_holder = DataHolder(mega_sample_length=max_sample_length, train_val_test_split=(.6,.2,.2), path=path, max_num_city=max_num_city)
    print(data_holder.cities)

    def get_dataloaders(sample_length, strides):
        train_dataset, val_dataset, test_dataset = data_holder.get_datasets_forecast(sample_length, strides=strides, one_city=one_city)
        train_dataloader = DataLoader(train_dataset, shuffle=True, drop_last=True, batch_size=train_batch)
        val_dataloader   = DataLoader(  val_dataset, shuffle=True, drop_last=True, batch_size= test_batch)
        test_dataloader  = DataLoader( test_dataset, shuffle=True, drop_last=True, batch_size= test_batch)
        return train_dataloader, val_dataloader, test_dataloader

    ## On commence par des s&quences courtes (3) :
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(10, strides=[5,10,10])
    if train_dataloader.__len__()==0 or test_dataloader.__len__()==0:
        raise AttributeError

    ## network definition :
    if not one_city:
        inDim = 3 + data_holder.get_num_classes() # température * month * day * hour
    else:
        inDim = 3 + 1
    outDim = inDim
    hidenDim = 35#2*outDim
    fc_layers = [] #[2*outDim]
    net = SequenceForecaster(inDim=inDim, hidenDim=hidenDim, outDim=outDim, fc_layers=fc_layers)
    loss_func = nn.MSELoss()

    ## optimizer :
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.005, betas=(.9, .999), eps=10**-8)
    #optimizer = torch.optim.SGD(params=net.parameters() , lr=.01, momentum=.9)

    ## display params :
    N = 10000
    loss_freq      = 1 # in number of epochs
    save_model_freq= 1
    histogram_freq = 1
    test_freq      = 1

    ## Savers :
    ONLY_SAVE_BEST_MODEL = True
    writer = SummaryWriter()
    MODEL_SAVE_PATH = 'best_model.pkl'
    print(MODEL_SAVE_PATH)
    if os.path.isfile(MODEL_SAVE_PATH):
        with open(MODEL_SAVE_PATH, 'rb') as f:
            SAVED_MODEL = pkl.load(f)
        net.load_state_dict(SAVED_MODEL['params'], strict=True)
        optimizer.load_state_dict(SAVED_MODEL['optimizer'])
        print('model loaded from "{}"\nepoch = {}, test loss = {}\n'.format(MODEL_SAVE_PATH, SAVED_MODEL['epoch'], SAVED_MODEL['loss_test']))
    else:
        SAVED_MODEL = {'loss_test':None, 'epoch':0}
        print('no best model found at "{}"\n'.format(MODEL_SAVE_PATH))


    ## Training loop :
    for i in tqdm(range(SAVED_MODEL['epoch'], N), desc='epochs'):
        ## test network :
        if i%test_freq==0 or i%save_model_freq==0:
            loss_test = validation_regress(net, test_dataloader, loss_func)
            writer.add_scalars('loss',     {'test':     loss_test}, i)
            writer.flush()
        ## save best model :
        if i%save_model_freq==0 and (not ONLY_SAVE_BEST_MODEL or SAVED_MODEL['loss_test'] is None or SAVED_MODEL['loss_test']>loss_test):
            SAVED_MODEL['epoch']     = i
            SAVED_MODEL['loss_test'] = loss_test
            SAVED_MODEL['params']    =       net.state_dict()
            SAVED_MODEL['optimizer'] = optimizer.state_dict()
            with open(MODEL_SAVE_PATH, 'wb') as f:
                pkl.dump(SAVED_MODEL, f)
            SAVED_MODEL['params'] = None # useless to keep the weights in memory
            print('\nepoch {} : new model saved\n'.format(i))

        ## train (one epoch) :
        optimizer.zero_grad()
        loss_train = 0.
        num_batchs = 0.
        for x, y_target in tqdm(train_dataloader, desc='train'):
            y = net(x)
            loss = loss_func(y, y_target)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            num_batchs += 1

        if i%loss_freq==0:
            loss_train /= num_batchs
            writer.add_scalars('loss',     {'train':     loss_train}, i)

        if i%histogram_freq==0:
            ## On enregistre les gradients à différentes couches pour constater si ils se propagent bien ou non :
            writer.add_histogram('weights/fi', net.rnn.fi.weight,      i)
            writer.add_histogram('weights/fh', net.rnn.fh.weight,      i)
            writer.add_histogram(  'grads/fi', net.rnn.fi.weight.grad, i)
            writer.add_histogram(  'grads/fh', net.rnn.fh.weight.grad, i)
            writer.add_histogram('outputs', y, i)

    writer.close()
