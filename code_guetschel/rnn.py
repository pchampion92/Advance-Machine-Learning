import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, inDim, hidenDim):
        super().__init__()
        self.inDim    = inDim
        self.hidenDim = hidenDim
        self.fi = nn.Linear(   inDim, hidenDim, bias=False)
        self.fh = nn.Linear(hidenDim, hidenDim, bias=True)
        self.activation = nn.Sigmoid()

    def one_step(self, x, h):
        x = self.fi(x)
        h = self.fh(h)
        return self.activation(x + h)

    def forward(self, X, h, many_to_many=False): ##X :seq_len*batch*indim, h:batch*hidenDim
        hiddens = [h]
        for i in range(X.shape[0]):
            hiddens.append(self.one_step(X[i], hiddens[i]))
        if many_to_many:
            return torch.stack(hiddens[1:], 0)
        return hiddens[-1]
