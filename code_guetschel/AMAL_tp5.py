import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import gzip

from torch_utils import NN, validation
from tp5_preprocess import TextDataset


##############################################################################
#### functions
##############################################################################

def staticbest_prediction(dataloader, labels):
    m, argm = None, None
    for l in labels:
        ll  = torch.tensor(l)
        acc = torch.tensor(0)
        for _,y in dataloader:
            acc += (y==ll).sum()
        if m is None or acc>m:
            m, argm = acc, ll
    print(m)
    return argm

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
def print_activation(token_list, interval, c, label):
    t = list(token_list)
    l = len(t)
    start = bcolors.OKBLUE if c==1 else bcolors.WARNING
    end   = bcolors.ENDC
    t.insert(min(l,interval[1]), end)
    t.insert(min(l,interval[0]), start)
    sentance = ''.join(t).replace('▁', ' ')
    correct = (bcolors.OKGREEN+'CORRECT' if c==label else bcolors.FAIL+'WRONG  ')+bcolors.ENDC
    print("true label : {} {} -> {}".format(label, correct, sentance))

def get_vocab(path='./wp1000.vocab'):
    with open(path, 'r') as f:
        lines = f.readlines()
    vocab = np.array([l.split('\t')[0] for l in lines])
    return vocab
def decode(seq, vocab):
    return [vocab[i] for i in seq]

##############################################################################
#### Modules
##############################################################################

class OneHot(nn.Module):
    def __init__(self, num_classes, dim):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
    def forward(self, x):
        x = nn.functional.one_hot(x, num_classes=self.num_classes)
        x = x.float()
        x = x.transpose(-1, self.dim)
        return x

class CNN1dClassifier(nn.Module):
    def __init__(self, inDim, embedDim, layers_desc, outDim):
        super().__init__()
        self.embedding = nn.Embedding(inDim, embedDim)
        self.layers = layers_desc
        layers_list = []
        last_dim = embedDim
        for (width, stride, dim) in layers_desc:
            layers_list.append( nn.Conv1d(last_dim, dim, width, stride=stride) )
            layers_list.append( nn.ReLU())
            last_dim = dim
        self.cnn = nn.Sequential(*layers_list)
        #self.classifier = nn.Linear(last_dim, outDim)
    def forward(self, x):
        x = self.embedding(x)
        x.transpose_(-1,-2)
        x = self.cnn(x)
        x, _ = torch.max(x, dim=-1)
        #x = self.classifier(x)
        return x
    def get_activations(self, x):
        x = self.embedding(x)
        x.transpose_(-1,-2)
        x = self.cnn(x)
        x, a = torch.max(x, dim=-1)
        a = np.stack((a.numpy(), a.numpy()+1), axis=-1) ## activations : intervals of one element
        for (width, stride, _) in self.layers:
            a = a*stride
            a = a + [0,width-1]
        #x = self.classifier(x)
        return x, a



##############################################################################
#### main
##############################################################################

if __name__=="__main__":
    from torch.utils.tensorboard import SummaryWriter
    import os
    import pickle as pkl

    num_tokens = 1000

    outDim = 2
    ## Parameters :
    args = {
        'batch_size':      500,
        'batch_size_test': 500,
        'embedDim':        30,
        'outDim':          outDim,
        'layers': [(3,1,20), (3,2,10), (2,1,outDim)],
    }

    ## Load data :
    print('loading data..')
    train_path = "train-{}.pth".format(num_tokens)
    test_path  = "test-{}.pth".format(num_tokens)
    with gzip.open(train_path, 'rb') as f:
        train_ds = torch.load(f)
    with gzip.open( test_path, 'rb') as f:
        test_ds  = torch.load(f)
    train_dataloader = DataLoader(train_ds, collate_fn=TextDataset.collate, shuffle=True, batch_size=args['batch_size'])
    test_dataloader  = DataLoader( test_ds, collate_fn=TextDataset.collate, shuffle=True, batch_size=args['batch_size_test'])
    print('data loaded')

    ## Compute static best baseline :
    print('\ncomputing static best..')
    #staticbest_label = torch.tensor(staticbest_prediction(train_dataloader, [0,1]))
    staticbest_label = torch.tensor(0)
    print('static best = {}'.format(staticbest_label))
    good, all = torch.tensor(0), torch.tensor(0)
    for _,y in test_dataloader:
        good += (y==staticbest_label).sum()
        all  += y.shape[0]
    staticbest_score = good.float()/all.float()
    print('staticbest_score = {} : {}/{}\n'.format(staticbest_score, good, all))

    ## Network :
    net = CNN1dClassifier(inDim=num_tokens, embedDim=args['embedDim'], layers_desc=args['layers'], outDim=args['outDim'])
    loss_func = nn.CrossEntropyLoss()

    vocab = get_vocab()

    print(len(train_dataloader), len(test_dataloader))

    ## optimizer :
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.005, betas=(.9, .999), eps=10**-8)
    #optimizer = torch.optim.SGD(params=net.parameters() , lr=.01, momentum=.9)

    ## display params :
    N = 10000
    loss_freq      = 200 # in number of epochs
    save_model_freq= 200
    histogram_freq = 400
    test_freq      = 200
    visual_test_freq=400
    num_lines      = 20

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
    loss_train = 0.
    num_batchs = 0.
    for epoch in tqdm(range(SAVED_MODEL['epoch'], N), desc='epochs'):
        for j, (x, y_target) in enumerate(tqdm(train_dataloader, desc='train')):
        ## train (one epoch) :
            i = epoch*len(train_dataloader) + j ## step num
            optimizer.zero_grad()
            net.zero_grad()
            y = net(x)
            loss = loss_func(y, y_target)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            num_batchs += 1

            if i%loss_freq==0:
                loss_train /= num_batchs
                writer.add_scalars('loss', {'train': loss_train}, i)
                loss_train, num_batchs = 0., 0.

            ## test network :
            if i%test_freq==0 or i%save_model_freq==0:
                loss_test, accuracy_test = validation(net, test_dataloader, loss_func)
                writer.add_scalars('loss',     {'test':     loss_test}, i)
                writer.add_scalars('accuracy', {'test': accuracy_test}, i)
                writer.flush()
            ## save best model :
            if i%save_model_freq==0 and (not ONLY_SAVE_BEST_MODEL or SAVED_MODEL['loss_test'] is None or SAVED_MODEL['loss_test']>loss_test):
                SAVED_MODEL['epoch']     = epoch
                SAVED_MODEL['loss_test'] = loss_test
                SAVED_MODEL['params']    =       net.state_dict()
                SAVED_MODEL['optimizer'] = optimizer.state_dict()
                with open(MODEL_SAVE_PATH, 'wb') as f:
                    pkl.dump(SAVED_MODEL, f)
                SAVED_MODEL['params'] = None # useless to keep the weights in memory
                print('\nepoch {} : new model saved\n'.format(i))

            if i%visual_test_freq==0:
                y,a = net.get_activations(x)
                out = torch.argmax(y, dim=-1)
                for k in range(num_lines):
                    seq, yy, yyt, aa = x[k].numpy(),out[k].numpy(), y_target[k].numpy(),a[k]
                    ## remove the padding at the end :
                    i = len(seq)
                    while i>0 and seq[i-1]==0:
                        i-=1
                    s = decode(seq[:i], vocab=vocab)
                    print_activation(s, aa[yy], yy, yyt)

            if i%histogram_freq==0:
                ## On enregistre les gradients à différentes couches pour constater si ils se propagent bien ou non :
                # writer.add_histogram('weights/fi', net.rnn.fi.weight,      i)
                # writer.add_histogram('weights/fh', net.rnn.fh.weight,      i)
                # writer.add_histogram(  'grads/fi', net.rnn.fi.weight.grad, i)
                # writer.add_histogram(  'grads/fh', net.rnn.fh.weight.grad, i)
                writer.add_histogram('outputs', y, i)

    writer.close()
