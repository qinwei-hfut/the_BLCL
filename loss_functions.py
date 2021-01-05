import torch.nn.functional as F
import torch
import torch.nn as nn
import pdb


class Soft_CE_loss(torch.nn.Module):
    def __init__(self):
        super(Soft_CE_loss,self).__init__()

    def forward(self, output, soft_target):
        return -torch.mean(torch.sum(F.log_softmax(output, dim=1) * soft_target, dim=1))

class CE_loss(torch.nn.Module):
    def __init__(self):
        super(CE_loss,self).__init__()

    def forward(self, output, target):
        return F.cross_entropy(output,target)

class MAE_one_hot_loss(torch.nn.Module):
    def __init__(self):
        super(MAE_one_hot_loss,self).__init__()
        # self.num_classes = num_classes
    
    def forward(self,output,target):
        output = F.softmax(output,dim=1)
        target_one_hot = F.one_hot(target,num_classes=output.size(1))
        return F.l1_loss(output,target_one_hot)


class MSE_one_hot_loss(torch.nn.Module):
    def __init__(self):
        super(MSE_one_hot_loss,self).__init__()

    def forward(self,output,target):
        output = F.softmax(output,dim=1)
        target_one_hot = F.one_hot(target,num_classes=output.size(1))
        return F.mse_loss(output,target_one_hot)

def Taylor_ce_loss_1(output,target):
    output = F.softmax(output,dim=1)
    error = 1-torch.gather(output,1,target.view(-1,1))
    return torch.mean(error)

def Taylor_ce_loss_2(output,target):
    output = F.softmax(output,dim=1)
    error = 1-torch.gather(output,1,target.view(-1,1))
    return torch.mean(error+torch.mul(error,error)/2)


class CE_MAE_loss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(CE_MAE_loss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss()

    # def MAE_loss(self,,output,target):
    #     output = F.softmax(output,dim=1)
    #     target_one_hot = F.one_hot(target,num_classes=output.size(1))
    #     target_one_hot = torch.clamp(target_one_hot,min=1e-4,max=1.0)
    #     return F.l1_loss(output,target_one_hot)

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # MAE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        # rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        MAE = F.l1_loss(pred,label_one_hot)

        # Loss
        # loss = self.alpha * ce + self.beta * rce.mean()
        loss = self.alpha * ce + self.beta * MAE
        return loss


class CE_LS_loss(torch.nn.Module):
    def __init__(self,alpha=0.9):
        super(CE_LS_loss,self).__init__()
        self.alpha = alpha
    
    def forward(self,preds,labels):
        num_classes = preds.size(1)
        # one_hot_label = (torch.zeros(len(labels), num_classes)+(1-alpha)/num_classes).scatter_(1, target.view(-1,1), 1) 
        LS_labels = (F.one_hot(labels, num_classes).float()*self.alpha+(1-self.alpha)/num_classes).to('cuda')
        return -torch.mean(torch.sum(F.log_softmax(preds, dim=1) * LS_labels, dim=1))


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


class Mixed_loss(torch.nn.Module):
    def __init__(self, alpha_ce, alpha_rce, alpha_mae, alpha_mse):
        super(Mixed_loss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha_ce = nn.ParameterList(map(nn.Parameter,nn.Parameter(torch.tensor(alpha_ce), requires_grad=True).cuda()))
        self.alpha_rce = nn.ParameterList(map(nn.Parameter,nn.Parameter(torch.tensor(alpha_rce), requires_grad=True).cuda()))
        self.alpha_mse = nn.ParameterList(map(nn.Parameter,nn.Parameter(torch.tensor(alpha_mse), requires_grad=True).cuda()))
        self.alpha_mae = nn.ParameterList(map(nn.Parameter,nn.Parameter(torch.tensor(alpha_mae), requires_grad=True).cuda()))

        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):

        num_classes = pred.size(1)

        # CCE
        ce = self.cross_entropy(pred, labels)

        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, num_classes).float().to(self.device)

        # MAE
        mae = F.l1_loss(pred,label_one_hot)

        # MSE
        mse = F.mse_loss(pred,label_one_hot)

        # RCE
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1)).mean()

        # Loss
        loss = self.alpha_ce * ce + self.alpha_rce * rce + self.alpha_mae * mae + self.alpha_mse * mse
        return loss