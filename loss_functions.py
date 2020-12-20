import torch.nn.functional as F
import torch

def ce_loss(output, target):
    return F.cross_entropy(output,target)

def soft_ce_loss(output, soft_target):
    return -torch.mean(torch.sum(F.log_softmax(output, dim=1) * soft_target, dim=1))

def MAE_loss(output, target):
    target = torch.zeros(len(target), 10).scatter_(1, target.view(-1,1), 1)
    return torch.nn.L1Loss(output,target)

def MSE_loss(output,target):
    target = torch.zeros(len(target), 10).scatter_(1, target.view(-1,1), 1)
    return F.mse_loss(output,target)

