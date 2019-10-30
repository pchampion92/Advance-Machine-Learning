# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
# from datamaestro import prepare_dataset
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import tqdm
from datamaestro import prepare_dataset

############
## Models ##
############

class MyDataset(Dataset):
    def __init__(self, prepared_dataset, data_name, target_name):
        self.data    = prepared_dataset.files[  data_name].data()
        self.targets = prepared_dataset.files[target_name].data()
        self.dataset_len = len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].flatten().astype(np.float32) / 255
        return (image, self.targets[idx])

    def __len__(self):
        return self.dataset_len


class Autoencoder(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.lin_encode = nn.Linear(input_shape, output_shape)
        self.w = self.lin_encode.weight
        self.lin_decode = lambda x: nn.functional.linear(x, self.w.t())
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lin_encode(x)
        x = self.relu(x)
        x = self.lin_decode(x)
        x = self.sigmoid(x)
        return x

if __name__=="__main__":

    ######################
    ## Creating dataset ##
    ######################

    ## training params :
    batch_size = 128
    test_batch_size = 1024
    lr = 10**-4
    momentum = .9
    N = 10000
    ## converge au bout d'une trentaine d'epochs avec ces paramètres et opti SGD

    ## net params :
    input_shape = 784
    output_shape = 10

    net = Autoencoder(input_shape, output_shape)
    loss_func = nn.BCELoss()
    optimizer = torch.optim.SGD(params=net.parameters() , lr=lr, momentum=momentum)
    #optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, betas=(.9, .999), eps=10**-8)

    ## Dataset
    prepared_dataset = prepare_dataset('com.lecun.mnist')
    dataset_train = DataLoader(MyDataset(prepared_dataset, 'train/images', 'train/labels'), shuffle=True, drop_last=True, batch_size=batch_size)
    dataset_test  = DataLoader(MyDataset(prepared_dataset,  'test/images',  'test/labels'), shuffle=True, drop_last=True, batch_size=test_batch_size)

    ## Writer options :
    writer = SummaryWriter()
    num_images_to_save = 10
    loss_freq      = 1 # num of epochs
    histogram_freq = 5
    test_freq      = 5

    ## Training loop :
    print("\n-----\nlr = {}\nmomentum = {}\nnum_epochs = {}\n-----".format(lr, momentum, N))
    for i in tqdm.tqdm(range(N), desc='epochs'):
        if i%test_freq==0:
            with torch.no_grad():
                loss_test = 0
                num_batchs = 0
                for x,_ in tqdm.tqdm(dataset_test, desc='test'):
                    x = x.float()
                    out_test = net.forward(x)
                    loss_test += loss_func(out_test, x)
                    num_batchs += 1
                loss_test = loss_test/num_batchs
            writer.add_scalars('loss', {'test': loss_test}, i)
            writer.add_images('images/origial',                x[:num_images_to_save].view(-1, 28, 28, 1), i, dataformats='NHWC')
            writer.add_images('images/encoded_decoded', out_test[:num_images_to_save].view(-1, 28, 28, 1), i, dataformats='NHWC')

        if i%histogram_freq==0:
            writer.add_histogram('weights', net.w, i)
            ## Sometimes i get the error "ValueError: The histogram is empty, please file a bug report."

        optimizer.zero_grad()
        loss_train = 0
        num_batchs = 0
        for x,_ in tqdm.tqdm(dataset_train, desc='train'):
            x = x.float()
            out_train = net.forward(x)
            loss = loss_func(out_train, x)
            loss.backward()
            optimizer.step()
            loss_train += loss
            num_batchs += 1
        loss_train = loss_train/num_batchs

        if i%loss_freq==0:
            writer.add_scalars('loss', {'train': loss_train}, i)
            writer.flush()

    writer.close()
