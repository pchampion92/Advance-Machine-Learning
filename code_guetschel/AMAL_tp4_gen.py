import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from rnn import RNN
from torch_utils import  NN, validation_regress
from AMAL_tp4_datasets import SpeachHolder


##############################################################################
#### Networks
##############################################################################

## entrées sous forme length × batch × dim

class SequenceGenerator(nn.Module):
    def __init__(self, inDim, embedDim, hidenDim, fc_layers, outDim):
        super().__init__()
        self.inDim, self.embedDim, self.hidenDim = inDim, embedDim, hidenDim

        self.embedding = nn.Embedding(inDim, embedDim)
        self.rnn = RNN(embedDim, hidenDim)
        self.fc = NN(hidenDim, outDim, fc_layers)

    def forward(self, x):      ## x : batch*seq_len*inDim
        x = self.embedding(x)
        h = - torch.ones(self.hidenDim)
        x = x.view(x.shape[0], x.shape[1], self.embedDim)
        x.transpose_(0,1)  ## now x : seq_len*batch*inDim
        x = self.rnn(x, h, many_to_many=True)
        x = self.fc(x)
        x.transpose_(0,1)  ## now x : batch*seq_len*inDim
        return x

    def generate(self, initial, length, max=False):
        hiddens = [- torch.ones(self.hidenDim)]
        seq = [i.item() for i in initial]
        for i in range(length-initial.shape[0]+1):
            hiddens.append(self.rnn(self.embedding(torch.tensor([seq[-1]])), hiddens[-1], many_to_many=False))
            probas = nn.functional.softmax(self.fc(hiddens[-1]))
            if max:
                x = torch.argmax(probas).view(1)
            else:
                x = torch.multinomial(probas, 1)
            seq.append(x.item())
        return seq


##############################################################################
#### main
##############################################################################

if __name__=="__main__":
    ## imports :
    from torch.utils.tensorboard import SummaryWriter
    import os
    import pickle as pkl

    ## load dataset temperatures :
    path="./trump_full_speech.txt"
    train_batch = 500
    test_batch  = 100

    data_holder = SpeachHolder(train_val_test_split=(.6,.2,.2), path=path)

    def get_dataloaders(sample_length, strides):
        train_dataset, val_dataset, test_dataset = data_holder.get_datasets(sample_length, strides=strides)
        train_dataloader = DataLoader(train_dataset, shuffle=True, drop_last=True, batch_size=train_batch)
        val_dataloader   = DataLoader(  val_dataset, shuffle=True, drop_last=True, batch_size= test_batch)
        test_dataloader  = DataLoader( test_dataset, shuffle=True, drop_last=True, batch_size= test_batch)
        return train_dataloader, val_dataloader, test_dataloader

    ## On commence par des s&quences courtes (3) :
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(50, strides=[5,50,50])
    if train_dataloader.__len__()==0 or test_dataloader.__len__()==0:
        raise AttributeError

    ## network definition :
    inDim = data_holder.get_num_classes() # encodage one_hot des caractères
    outDim = inDim
    embedDim = 50
    hidenDim = 50#2*outDim
    fc_layers = [] #[2*outDim]
    net = SequenceGenerator(inDim=inDim, embedDim=embedDim, hidenDim=hidenDim, outDim=outDim, fc_layers=fc_layers)
    ll = nn.CrossEntropyLoss()
    loss_func = lambda y,yt: ll(y.transpose(1,-1), yt)

    ## optimizer :
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.005, betas=(.9, .999), eps=10**-8)
    #optimizer = torch.optim.SGD(params=net.parameters() , lr=.01, momentum=.9)

    ## display params :
    N = 10000
    loss_freq      = 1 # in number of epochs
    save_model_freq= 1
    histogram_freq = 1
    test_freq      = 1
    generate_freq  = 2

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

    ## generate  a sequence
    with torch.no_grad():
        initial = data_holder.string2code('I, Donald Trump')
        seq = net.generate(initial, 100, max=False)
        print('\niter{} :\n{}\n'.format(SAVED_MODEL['epoch'], data_holder.code2string(seq)))


    ## Training loop :
    for i in tqdm(range(SAVED_MODEL['epoch'], N), desc='epochs'):
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

        if i%histogram_freq==0:
            ## On enregistre les gradients à différentes couches pour constater si ils se propagent bien ou non :
            writer.add_histogram('weights/fi', net.rnn.fi.weight,      i)
            writer.add_histogram('weights/fh', net.rnn.fh.weight,      i)
            writer.add_histogram(  'grads/fi', net.rnn.fi.weight.grad, i)
            writer.add_histogram(  'grads/fh', net.rnn.fh.weight.grad, i)
            writer.add_histogram('outputs', y, i)

        if i%generate_freq==0:
            with torch.no_grad():
                initial = data_holder.string2code('I, Donald Trump, ')
                seq = net.generate(initial, 1000)
                print('\niter{} :\n{}\n'.format(i, data_holder.code2string(seq)))


    writer.close()


##  16 epoch :
# I, Donald Trump, Mcano t'vestowetesiso aco m or wer oro ll bil b iersu  sere fm
# ller s s alye Amo men slyo hre ttrol f to trer a mt. il te d f dos ll Weweser
# Caco durersarigo g lyorero sur l har torinto topl f r te o -he tr thecosenttor
# drede b I tr bowetem bsinyojuo hl  t'sts rale sexlyr o Ir tor, cer bto yes otrm
# tr to e o t'sil to tamtrar al l dem exgoraperetell tateretoususto wigerkese sor
# tos th bese helatom sle ico ser thisme-exthopll tse " f Amess -ve l tr mo Am in
# atto s Urigtes bo al bitro anger es then tr to y Chinen ttpl Ito en ullareivene
# resmio vene ore terecesingeoreres ye tormme  -tll ceses hethou po asor te s evely
# bsive e tou the tthcarenos oristo Isdres t ste o i pril   ark oror avor cresur
# iseseeongese tto Sbo ply Amn's p copore orsire e p prtonor y rther h cerapllye
# f ace cigerehe oro yo and, arou tengor Gtbe osear  icelesse ano be l Ug bkeS thes
# pobere, hesto bmeeret r tsor ak f gobo hocelly d e Ptro ar b toru asic alerometowe
# Woll nu tosusose Ug Ul  f icermes or
