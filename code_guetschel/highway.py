import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from collections import OrderedDict
import tqdm


class HighwayBlock(nn.Module):
    def __init__(self, H, T, C):
        super().__init__()
        self.H = H
        self.T = T
        self.C = C
    def forward(self, x):
        return self.H(x)*self.T(x) + x*self.C(x)

class HighwaySimpleBlock(nn.Module):
    def __init__(self, H, T):
        super().__init__()
        self.H = H
        self.T = T
        #### C = 1 - T
    def forward(self, x):
        return (self.H(x) - x)*self.T(x) + x

class SigmoidGate(nn.Module):
    def __init__(self, input_size, initial_bias=None):
        super().__init__()
        self.fc = nn.Linear(input_size, input_size)
        self.sig = nn.Sigmoid()
        if initial_bias is not None:
            params = list(self.fc.parameters())
            params[1] = initial_bias
    def forward(self, x):
        x = self.fc(x)
        x = self.sig(x)
        return x

class FC_ReLU(nn.Sequential):
    def __init__(self, input_size):
        super().__init__(
            nn.Linear(input_size, input_size),
            nn.ReLU()
        )
class FC_Tanh(nn.Sequential):
    def __init__(self, input_size):
        super().__init__(
            nn.Linear(input_size, input_size),
            nn.Tanh()
        )

class HighwayNet(nn.Sequential):
    def __init__(self, input_size, output_size, num_units=50, depth=10):
        blocks = OrderedDict()
        blocks['fc0'] = nn.Linear(input_size, num_units)

        initial_bias = -3
        for i in range(depth):
            b = HighwaySimpleBlock(FC_ReLU(num_units), SigmoidGate(num_units, initial_bias))
            block_name = 'block{}'.format(i)
            blocks[block_name] = b

        blocks['fc1'] = nn.Linear(num_units, output_size)
        blocks['softmax'] = nn.Softmax(dim=-1)
        
        super().__init__(blocks) ## initialize the Sequential module with the OrderedDict 'blocks'

## dataset related :
class ReshapeTransform(object):
    def __init__(self, new_size):
        self.new_size = new_size
    def __call__(self, img):
        return torch.reshape(img, self.new_size)

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

def compute_accuracy(y_pred, y_target):
    _, y_idx_pred = y_pred.max(-1)
    goods = torch.sum(torch.eq(y_idx_pred, y_target))
    all = y_target.shape[0]
    return goods, all

if __name__=="__main__":

    ## training params :
    batch_size = 128
    test_batch_size = 1024
    lr = 10**-3
    momentum = .9
    N = 10000

    ## net params :
    input_shape = 784
    output_shape = 10
    num_units = 50
    depth = 9

    net = HighwayNet(input_shape, output_shape, num_units, depth)
    optimizer = torch.optim.SGD(params=net.parameters() , lr=lr, momentum=momentum)
    #optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, betas=(.9, .999), eps=10**-8)
    loss_func = nn.NLLLoss()

    ## Dataset
    mnist_tr = transforms.Compose([
            transforms.ToTensor(),
            ReshapeTransform((-1,)),
        ])
    dataset_train = DataLoader(MNIST(root="~/datasets/", train=True,  transform=mnist_tr), shuffle=True, drop_last=True, batch_size=batch_size)
    dataset_test  = DataLoader(MNIST(root="~/datasets/", train=False, transform=mnist_tr), shuffle=True, drop_last=True, batch_size=test_batch_size)

    ## Writer options :
    writer = SummaryWriter(flush_secs=10)

    # Aucun graph ne s'affiche ...
    # image,_ = next(iter(dataset_test))
    # writer.add_graph(net, image)
    # writer.flush()

    loss_freq      = 1 # num of epochs
    histogram_freq = 1
    test_freq      = 1

    print("\n-----\nlr = {}\nnum_epochs = {}\n-----".format(lr, N))

    ## Training loop :
    for i in tqdm.tqdm(range(N), desc='epochs'):
        if i%test_freq==0:
            with torch.no_grad():
                loss_test = 0.
                num_batchs = 0.
                accuracy_test = 0.
                all_elements = 0.
                for x, y_target in tqdm.tqdm(dataset_test, desc='test'):
                    y = net(x)
                    loss_test += loss_func(y, y_target)
                    num_batchs += 1
                    good, all = compute_accuracy(y, y_target)
                    accuracy_test += good
                    all_elements += all
                loss_test /= num_batchs
                accuracy_test = accuracy_test.float() / all_elements
            writer.add_scalars('loss',     {'test':     loss_test}, i)
            writer.add_scalars('accuracy', {'test': accuracy_test}, i)
            writer.flush()

        optimizer.zero_grad()
        loss_train = 0.
        num_batchs = 0.
        accuracy_train = 0.
        all_elements = 0.
        for x, y_target in tqdm.tqdm(dataset_train, desc='train'):
            y = net(x)
            loss = loss_func(y, y_target)
            loss.backward()
            optimizer.step()
            loss_train += loss
            num_batchs += 1
            good, all = compute_accuracy(y, y_target)
            accuracy_train += good
            all_elements += all
        loss_train /= num_batchs
        accuracy_train = accuracy_train.float() / all_elements

        if i%loss_freq==0:
            writer.add_scalars('loss',     {'train':     loss_train}, i)
            writer.add_scalars('accuracy', {'train': accuracy_train}, i)

        if i%histogram_freq==0:
            ## On enregistre les gradients à différentes couches pour constater si ils se propagent bien ou non :
            writer.add_histogram('gradients/first_block', net[1         ].H[0].weight.grad, i)
            writer.add_histogram('gradients/mid_block',   net[depth//2+1].H[0].weight.grad, i)
            writer.add_histogram('gradients/last_block',  net[depth     ].H[0].weight.grad, i)

    writer.close()
