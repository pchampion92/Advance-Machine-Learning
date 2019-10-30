import torch
from torch import nn
from tqdm import tqdm


##############################################################################
#### modules :
##############################################################################

class NN(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(NN, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))
    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x)
            x = self.layers[i](x)
        return x


##############################################################################
#### Functions
##############################################################################


def validation(net, dataloader, loss_fun):
        with torch.no_grad():
            loss       = 0.
            num_batchs = 0.
            accuracy   = 0.
            all        = 0.
            for x, y_target in tqdm(dataloader, desc='validation'):
                y = net(x)
                l = loss_fun(y, y_target)
                loss       += l.item()
                num_batchs += 1
                accuracy   += torch.sum(torch.argmax(y, dim=-1)==y_target).item()
                all        += y.shape[0]
            if num_batchs==0 or all==0:
                return None, None
            loss /= float(num_batchs)
            accuracy /= float(all)
            return loss, accuracy

def validation_regress(net, dataloader, loss_fun):
        with torch.no_grad():
            loss       = 0.
            num_batchs = 0.
            for x, y_target in tqdm(dataloader, desc='validation'):
                y = net(x)
                l = loss_fun(y, y_target)
                loss       += l.item()
                num_batchs += 1
            if num_batchs==0 or all==0:
                return None
            loss /= float(num_batchs)
            return loss
