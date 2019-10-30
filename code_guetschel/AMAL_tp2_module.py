import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tqdm


class NetModule(torch.nn.Module):
    def __init__(self, feature_size, internal_size):
        super(NetModule, self).__init__()
        self.f1 = torch.nn.Linear(feature_size, internal_size)
        self.tanh = torch.nn.Tanh()
        self.f2 = torch.nn.Linear(internal_size, internal_size2)

    def forward(self, x):
        x = self.f1(x)
        x = self.tanh(x)
        x = self.f2(x)
        return x


if __name__=="__main__":

    ###################################
    ## Testnig on the boston dataset ##
    ###################################

    ## Pour telecharger le dataset Boston
    # ds=prepare_dataset("edu.uci.boston")
    # fields, data =ds.files.data()
    from sklearn.datasets import load_boston
    boston_dataset = load_boston()
    print('dataset loaded')
    print(boston_dataset.keys())
    all_data =      boston_dataset['data']
    all_target =    boston_dataset['target']
    feature_names = boston_dataset['feature_names']
    dataset_size = all_data.shape[0]
    feature_size = feature_names.shape[0]
    print(all_data.dtype, all_target.dtype)
    ## split dataset :
    p = 0.2
    test_mask = np.random.choice([True, False], dataset_size, p=(p, 1-p))
    with torch.no_grad():
        test_target =  torch.from_numpy(all_target[test_mask]).view(-1, 1)
        test_data =    torch.from_numpy(all_data[test_mask, :]).view(-1, 1, feature_size)
    train_target = torch.from_numpy(all_target[~test_mask]).view(-1, 1)
    train_data =   torch.from_numpy(all_data[~test_mask, :]).view(-1, 1, feature_size)
    test_size = test_data.shape[0]
    train_size = train_data.shape[0]


    ## same parameters as in previous examples :
    lr = 10**-1
    momentum = 0
    N = 20
    internal_size = 10
    internal_size2 = 10

    net = NetModule(feature_size, internal_size)
    # net = torch.nn.Sequential(
    #     torch.nn.Linear(feature_size, internal_size),
    #     torch.nn.Tanh(),
    #     torch.nn.Linear(internal_size, internal_size2),
    #     )
    net.double()
    mse_loss = torch.nn.MSELoss(torch.float64)
    optimizer = torch.optim.SGD(params=net.parameters() , lr=lr, momentum=momentum)


    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    loss_freq = 1
    histogram_freq = 1

    print("\n-----\nlr = {}\nmomentum = {}\nnum_epochs = {}\n-----".format(lr, momentum, N))
    for i in tqdm.tqdm(range(N)):
        optimizer.zero_grad()
        with torch.no_grad():
            out_test = net.forward(test_data)
            loss_test = mse_loss(out_test, test_target)

        out_train = net.forward(train_data)
        loss_train = mse_loss(out_train, train_target)
        loss_train.backward()
        optimizer.step()

        ## Save log :
        if i%loss_freq==0:
            writer.add_scalars('loss', {'test': loss_test, 'train': loss_train}, i)
        if i%histogram_freq==0:
            pass
            #writer.add_histogram('weights', net.w, i)
            ## Sometimes i get the error "ValueError: The histogram is empty, please file a bug report."

    writer.close()
