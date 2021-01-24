import torch
import torch.nn
import torch.nn.functional as F

class Sigmoid(torch.nn.Module):
    def __init__(self):
        super(Sigmoid,self).__init__()

    def forward(self,x):
        return F.sigmoid(x)

class Tanh_1(torch.nn.Module):
    def __init__(self):
        super(Tanh_1,self).__init__()

    def forward(self,x):
        return F.tanh(x)+torch.tensor(1.0).cuda()