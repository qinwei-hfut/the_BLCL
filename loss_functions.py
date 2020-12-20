import torch.nn.functional as F
import torch
import pdb

def ce_loss(output, target):
    return F.cross_entropy(output,target)

def soft_ce_loss(output, soft_target):
    return -torch.mean(torch.sum(F.log_softmax(output, dim=1) * soft_target, dim=1))

def MAE_loss(output, target):
    output = F.softmax(output,dim=1)
    target = torch.zeros(len(target), 10).cuda().scatter_(1, target.view(-1,1), 1)
    return F.l1_loss(output,target)

def MSE_loss(output,target):
    output = F.softmax(output,dim=1)
    target = torch.zeros(len(target), 10).cuda().scatter(1, target.view(-1,1), 1)
    return F.mse_loss(output,target)

def Taylor_ce_loss_1(output,target):
    output = F.softmax(output,dim=1)
    return torch.mean(1-output[target])

def Taylor_ce_loss_2(output,target):
    output = F.softmax(output,dim=1)
    return torch.mean((1-output[target])+torch.mul((1-output[target]),(1-output[target]))/2)
