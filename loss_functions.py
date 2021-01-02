import torch.nn.functional as F
import torch
import pdb

# def ce_loss(output, target):
#     return F.cross_entropy(output,target)

def soft_ce_loss(output, soft_target):
    return -torch.mean(torch.sum(F.log_softmax(output, dim=1) * soft_target, dim=1))

class CE_loss(torch.nn.Module):
    def __init__(self):
        super(CE_loss,self).__init__()

    def forward(output, target):
        return F.cross_entropy(output,target)

class MAE_one_hot_loss(torch.nn.Module):
    def __init__(self,num_classes=10):
        super(MAE_one_hot_loss,self).__init__()
        # self.num_classes = num_classes
    
    def forward(self,output,target):
        output = F.softmax(output,dim=1)
        target_one_hot = F.one_hot(target,num_classes=output.size(1))
        return F.l1_loss(output,target_one_hot)


def MSE_loss(output,target):
    output = F.softmax(output,dim=1)
    # TODO
    target = torch.zeros(len(target), 10).cuda().scatter(1, target.view(-1,1), 1)
    return F.mse_loss(output,target)

def Taylor_ce_loss_1(output,target):
    output = F.softmax(output,dim=1)
    error = 1-torch.gather(output,1,target.view(-1,1))
    return torch.mean(error)

def Taylor_ce_loss_2(output,target):
    output = F.softmax(output,dim=1)
    error = 1-torch.gather(output,1,target.view(-1,1))
    return torch.mean(error+torch.mul(error,error)/2)

class SCE_loss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCE_loss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
