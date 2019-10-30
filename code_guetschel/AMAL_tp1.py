# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
# from datamaestro import prepare_dataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tqdm


############
## Models ##
############

class Context:
    """Very simplified context object"""
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    @staticmethod
    def forward(ctx, input, label):
        ctx.save_for_backward(input, label)
        out = (input - label)**2
        #print(input.shape, label.shape, out.shape)
        return out

    @staticmethod
    def backward(ctx,grad_output):
        input, label = ctx.saved_tensors
        g_inp = torch.matmul(grad_output.unsqueeze(-1), 2*(input - label).unsqueeze(-2)).squeeze(-1)
        g_lab = torch.matmul(grad_output.unsqueeze(-1), 2*(label - input).unsqueeze(-2)).squeeze(-1)
        return g_inp, g_lab

class Linear(Function):
    @staticmethod
    def forward(ctx, x, w, b):
        ## Calcul la sortie du module
        ctx.save_for_backward(x, w, b)
        output = torch.matmul(x, w.view(-1, 1)) + b
        return output.view(-1,1)

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        x,w,b = ctx.saved_tensors
        g_x = torch.matmul(grad_output.view(-1, 1, 1), w.view(1, -1))
        g_w = torch.matmul(grad_output.view(-1, 1, 1), x).squeeze(-2)
        g_b = grad_output
        #print(g_x.shape, g_w.shape, g_b.shape)
        return g_x, g_w, g_b


class Net:
    def __init__(self, feature_size):
        self.feature_size = feature_size
        self.w = torch.randn(feature_size, requires_grad=True, dtype=torch.float64)
        self.b = torch.randn(1, requires_grad=True, dtype=torch.float64)
        self.linear = Linear()
        self.ctx = Context()
        self.mse = MSE()
        self.ctx_mse = Context()

    def forward(self, x, target):
        output = self.linear.forward(self.ctx, x, self.w, self.b)
        loss = self.mse.forward(self.ctx_mse, output, target)
        loss_mean = torch.mean(loss, dim=0)[0]
        self.last_batch_size = x.shape[0]
        return output, loss, loss_mean

    def backpropagate(self, lr=.1):
        ones = torch.ones(self.last_batch_size, 1, requires_grad=True, dtype=torch.float64)
        y_grad, _ = self.mse.backward(self.ctx_mse, ones)
        _, grad_w, grad_b = self.linear.backward(self.ctx, y_grad)
        #print("\nw = \n{}grad_w = \n{}".format( self.w, torch.sum(grad_w, dim=0)))
        ## update weights :
        self.w = self.w - lr*torch.sum(grad_w, dim=0)
        self.b = self.b - lr*torch.sum(grad_b, dim=0)



if __name__=="__main__":

    #######################
    ## Testing functions ##
    #######################

    linear = Linear()
    ctx = Context()
    mse = MSE()
    ctx_mse = Context()

    batch_size = 5#10
    feature_size = 3#13

    torch.manual_seed(12)
    x = torch.randn(batch_size, 1, feature_size, requires_grad=True, dtype=torch.float64)
    w = torch.randn(feature_size, requires_grad=True, dtype=torch.float64)
    b = torch.randn(1, requires_grad=True, dtype=torch.float64)
    y = torch.randn(batch_size, 1, requires_grad=True, dtype=torch.float64)
    yy =torch.randn(batch_size, 1, requires_grad=True, dtype=torch.float64)

    ## Test forward backward
    yy = linear.forward(ctx, x, w, b)
    output = mse.forward(ctx_mse, yy, y)
    mse_grad, _ = mse.backward(ctx_mse, torch.ones(batch_size, 1, requires_grad=True, dtype=torch.float64))
    linear_grad = linear.backward(ctx, mse_grad)
    print('test forward, backward \tdone')

    ## Pour tester le gradient
    linear_check = Linear.apply
    torch.autograd.gradcheck(linear_check,(x, w, b))
    mse_check = MSE.apply
    torch.autograd.gradcheck(mse_check, (yy, y))
    print("gradcheck         \tdone")


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
    test_data =    torch.from_numpy(all_data[test_mask, :]).view(-1, 1, feature_size)
    test_target =  torch.from_numpy(all_target[test_mask]).view(-1, 1)
    train_data =   torch.from_numpy(all_data[~test_mask, :]).view(-1, 1, feature_size)
    train_target = torch.from_numpy(all_target[~test_mask]).view(-1, 1)
    test_size = test_data.shape[0]
    train_shape = train_data.shape[0]

    net = Net(feature_size)
    lr = 10**-9
    N = 10000

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    loss_freq = 10
    histogram_freq = 50

    print("\n-----\nlr = {}\nnum_epochs = {}\n-----".format(lr, N))
    for i in tqdm.tqdm(range(N)):
        #print('\n-- epoch {} --'.format(i))
        _, _, loss_test = net.forward(test_data, test_target)
        #print('test_loss = {}'.format(loss_test))

        _, _, loss_train = net.forward(train_data, train_target)
        #print('train_loss = {}'.format(loss_train))
        #print("avg(w) = {}, b = {}".format(torch.mean(net.w), net.b[0]))
        net.backpropagate(lr=lr)
        if i%loss_freq==0:
            writer.add_scalars('loss', {'test': loss_test, 'train': loss_train}, i)
        #writer.add_scalar('loss/test_all_iter', loss_test, i)
        if i%histogram_freq==0:
            pass
            #writer.add_histogram('weights', net.w, i)
            ## Sometimes i get the error "ValueError: The histogram is empty, please file a bug report."

    writer.close()
