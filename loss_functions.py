import torch.nn.functional as F
import torch

def ce_loss(output, target):
    return F.cross_entropy(output,target)

def soft_ce_lss(output, soft_target):
    return -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * soft_target, dim=1))