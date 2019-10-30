import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tqdm

from AMAL_tp1 import Linear, MSE



class NetOptimizer:
    def __init__(self, feature_size, lr, momentum=0):
        self.feature_size = feature_size
        self.w = torch.randn(feature_size, requires_grad=True, dtype=torch.float64)
        self.b = torch.randn(1, requires_grad=True, dtype=torch.float64)
        self.linear = Linear.apply
        self.mse = MSE.apply
        self.optimizer = torch.optim.SGD(params=(self.w, self.b), lr=lr, momentum=momentum)

    def forward(self, x, target, optimize=False):
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(optimize):
            output = self.linear(x, self.w, self.b)
            loss = self.mse(output, target)
            loss_mean = torch.mean(loss, dim=0)
        if optimize:
            loss_mean.backward()
            self.optimizer.step()
        return (output, loss, loss_mean)


if __name__=="__main__":

    linear = Linear.apply
    mse = MSE.apply

    #######################
    ## Testing functions ##
    #######################

    batch_size = 5#10
    feature_size = 3#13
    torch.manual_seed(12)
    x = torch.randn(batch_size, 1, feature_size, requires_grad=True, dtype=torch.float64)
    w = torch.randn(feature_size, requires_grad=True, dtype=torch.float64)
    b = torch.randn(1, requires_grad=True, dtype=torch.float64)
    y = torch.randn(batch_size, 1, requires_grad=True, dtype=torch.float64)
    yy =torch.randn(batch_size, 1, requires_grad=True, dtype=torch.float64)

    ## Pour tester le gradient
    torch.autograd.gradcheck(linear,(x, w, b))
    torch.autograd.gradcheck(mse, (yy, y))
    print("gradcheck \tdone")


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
    lr = 10**-9 * train_size
    momentum = 0
    N = 10000
    net = NetOptimizer(feature_size, lr=lr, momentum=momentum)

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    loss_freq = 10
    histogram_freq = 50

    print("\n-----\nlr = {}\nmomentum = {}\nnum_epochs = {}\n-----".format(lr, momentum, N))
    for i in tqdm.tqdm(range(N)):
        _, _, loss_test = net.forward(test_data, test_target, optimize=False)
        _, _, loss_train = net.forward(train_data, train_target, optimize=True)

        ## Save log :
        if i%loss_freq==0:
            writer.add_scalars('loss', {'test': loss_test, 'train': loss_train}, i)
        if i%histogram_freq==0:
            pass
            #writer.add_histogram('weights', net.w, i)
            ## Sometimes i get the error "ValueError: The histogram is empty, please file a bug report."

    writer.close()
