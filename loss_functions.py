import torch.nn.functional as F
import torch
import torch.nn as nn
import pdb


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Soft_CE_loss(torch.nn.Module):
    def __init__(self):
        super(Soft_CE_loss,self).__init__()

    def forward(self, output, soft_target):
        return -torch.mean(torch.sum(F.log_softmax(output, dim=1) * soft_target, dim=1))

class CE_loss(torch.nn.Module):
    def __init__(self,reduction="mean"):
        super(CE_loss,self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, output, target):
        return self.ce(output,target)

class MAE_one_hot_loss(torch.nn.Module):
    def __init__(self):
        super(MAE_one_hot_loss,self).__init__()
        # self.num_classes = num_classes
    
    def forward(self,output,target):
        output = F.softmax(output,dim=1)
        target_one_hot = F.one_hot(target,num_classes=output.size(1)).float()
        return F.l1_loss(output,target_one_hot)


class MSE_one_hot_loss(torch.nn.Module):
    def __init__(self):
        super(MSE_one_hot_loss,self).__init__()

    def forward(self,output,target):
        output = F.softmax(output,dim=1)
        target_one_hot = F.one_hot(target,num_classes=output.size(1)).float()
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
        self.num_classes = pred.size(1)
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
    def __init__(self, alpha_ce, alpha_rce, alpha_mae, alpha_mse,activation_type):
        super(Mixed_loss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha_ce = nn.Parameter(torch.tensor(alpha_ce), requires_grad=True)
        self.alpha_rce = nn.Parameter(torch.tensor(alpha_rce), requires_grad=True)
        self.alpha_mse = nn.Parameter(torch.tensor(alpha_mse), requires_grad=True)
        self.alpha_mae = nn.Parameter(torch.tensor(alpha_mae), requires_grad=True)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.activation = getattr(torch.nn,activation_type)()

    def forward(self, pred, labels):

        self.to(self.device)
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
        ce_weight,rce_weight,mae_weight,mse_weight = self.activation(self.alpha_ce),self.activation(self.alpha_rce),self.activation(self.alpha_mae),self.activation(self.alpha_mse)
        loss = ce_weight * ce + rce_weight * rce + mae_weight * mae + mse_weight * mse
        return loss


class Mixed_loss_rce_ce(torch.nn.Module):
    def __init__(self, alpha_ce, alpha_rce, alpha_mae, alpha_mse,activation_type):
        super(Mixed_loss_rce_ce, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha_ce = nn.Parameter(torch.tensor(alpha_ce), requires_grad=True)
        self.alpha_rce = nn.Parameter(torch.tensor(alpha_rce), requires_grad=True)
        self.alpha_mse = nn.Parameter(torch.tensor(alpha_mse), requires_grad=True)
        self.alpha_mae = nn.Parameter(torch.tensor(alpha_mae), requires_grad=True)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.activation = getattr(torch.nn,activation_type)()

    def forward(self, pred, labels):

        self.to(self.device)
        num_classes = pred.size(1)

        # CCE
        ce = self.cross_entropy(pred, labels)

        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, num_classes).float().to(self.device)
        # RCE
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1)).mean()

        # Loss
        ce_weight,rce_weight,mae_weight,mse_weight = self.activation(self.alpha_ce),self.activation(self.alpha_rce),self.activation(self.alpha_mae),self.activation(self.alpha_mse)
        loss = ce_weight * ce + rce_weight * rce
        return loss

class Mixed_loss_sample(torch.nn.Module):
    def __init__(self,activation_type):
        super(Mixed_loss_sample, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.activation = getattr(torch.nn,activation_type)()

    def forward(self, pred, labels, loss_weight_per_sample):

        self.to(self.device)
        num_classes = pred.size(1)
        batch_size = pred.size(0)

        # alpha_ce, alpha_mae, alpha_mse,alpha_rce = alpha_ce.view(batch_size,-1), alpha_mae.view(batch_size,-1), alpha_mse.view(batch_size,-1), alpha_rce.view(batch_size,-1)
        alpha_ce, alpha_mae, alpha_mse,alpha_rce = loss_weight_per_sample[:,0].view(batch_size), loss_weight_per_sample[:,1].view(batch_size), loss_weight_per_sample[:,2].view(batch_size), loss_weight_per_sample[:,3].view(batch_size)
        # CCE
        ce = self.cross_entropy(pred, labels)

        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, num_classes).float().to(self.device)
        # MAE
        mae = F.l1_loss(pred,label_one_hot,reduction='none').sum(dim=1)
        # MSE
        mse = F.mse_loss(pred,label_one_hot,reduction='none').sum(dim=1)
        # RCE
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        ce_weight,rce_weight,mae_weight,mse_weight = self.activation(alpha_ce),self.activation(alpha_rce),self.activation(alpha_mae),self.activation(alpha_mse)
        loss = ce_weight * ce + rce_weight * rce + mae_weight * mae + mse_weight * mse
        loss = loss.mean()
        return loss

class Mixed_loss_rce(torch.nn.Module):
    def __init__(self, alpha_ce, alpha_rce, alpha_mae, alpha_mse,activation_type):
        super(Mixed_loss_rce, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha_ce = nn.Parameter(torch.tensor(alpha_ce), requires_grad=True)
        self.alpha_rce = nn.Parameter(torch.tensor(alpha_rce), requires_grad=True)
        self.alpha_mse = nn.Parameter(torch.tensor(alpha_mse), requires_grad=True)
        self.alpha_mae = nn.Parameter(torch.tensor(alpha_mae), requires_grad=True)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.activation = getattr(torch.nn,activation_type)()

    def forward(self, pred, labels):

        self.to(self.device)
        num_classes = pred.size(1)

        # CCE
        ce = self.cross_entropy(pred, labels)

        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, num_classes).float().to(self.device)
        # MAE
        mae = F.l1_loss(pred,label_one_hot)
        # MSE
        mse = F.mse_loss(pred,label_one_hot)


        # Loss
        ce_weight,rce_weight,mae_weight,mse_weight = self.activation(self.alpha_ce),self.activation(self.alpha_rce),self.activation(self.alpha_mae),self.activation(self.alpha_mse)
        loss = ce_weight * ce + mae_weight * mae + mse_weight * mse
        return loss


class Mixed_loss_ce(torch.nn.Module):
    def __init__(self, alpha_ce, alpha_rce, alpha_mae, alpha_mse,activation_type):
        super(Mixed_loss_ce, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha_ce = nn.Parameter(torch.tensor(alpha_ce), requires_grad=True)
        self.alpha_rce = nn.Parameter(torch.tensor(alpha_rce), requires_grad=True)
        self.alpha_mse = nn.Parameter(torch.tensor(alpha_mse), requires_grad=True)
        self.alpha_mae = nn.Parameter(torch.tensor(alpha_mae), requires_grad=True)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.activation = getattr(torch.nn,activation_type)()

    def forward(self, pred, labels):

        self.to(self.device)
        num_classes = pred.size(1)

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
        ce_weight,rce_weight,mae_weight,mse_weight = self.activation(self.alpha_ce),self.activation(self.alpha_rce),self.activation(self.alpha_mae),self.activation(self.alpha_mse)
        loss = rce_weight * rce + mae_weight * mae + mse_weight * mse
        return loss

class Mixed_loss_mae(torch.nn.Module):
    def __init__(self, alpha_ce, alpha_rce, alpha_mae, alpha_mse,activation_type):
        super(Mixed_loss_mae, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha_ce = nn.Parameter(torch.tensor(alpha_ce), requires_grad=True)
        self.alpha_rce = nn.Parameter(torch.tensor(alpha_rce), requires_grad=True)
        self.alpha_mse = nn.Parameter(torch.tensor(alpha_mse), requires_grad=True)
        self.alpha_mae = nn.Parameter(torch.tensor(alpha_mae), requires_grad=True)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.activation = getattr(torch.nn,activation_type)()

    def forward(self, pred, labels):

        self.to(self.device)
        num_classes = pred.size(1)

        # CCE
        ce = self.cross_entropy(pred, labels)

        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, num_classes).float().to(self.device)

        # MSE
        mse = F.mse_loss(pred,label_one_hot)
        # RCE
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1)).mean()

        # Loss
        ce_weight,rce_weight,mae_weight,mse_weight = self.activation(self.alpha_ce),self.activation(self.alpha_rce),self.activation(self.alpha_mae),self.activation(self.alpha_mse)
        loss = ce_weight * ce + rce_weight * rce + mse_weight * mse
        return loss

class Mixed_loss_mse(torch.nn.Module):
    def __init__(self, alpha_ce, alpha_rce, alpha_mae, alpha_mse,activation_type):
        super(Mixed_loss_mse, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha_ce = nn.Parameter(torch.tensor(alpha_ce), requires_grad=True)
        self.alpha_rce = nn.Parameter(torch.tensor(alpha_rce), requires_grad=True)
        self.alpha_mse = nn.Parameter(torch.tensor(alpha_mse), requires_grad=True)
        self.alpha_mae = nn.Parameter(torch.tensor(alpha_mae), requires_grad=True)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.activation = getattr(torch.nn,activation_type)()

    def forward(self, pred, labels):

        self.to(self.device)
        num_classes = pred.size(1)

        # CCE
        ce = self.cross_entropy(pred, labels)

        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, num_classes).float().to(self.device)
        # MAE
        mae = F.l1_loss(pred,label_one_hot)
        # RCE
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1)).mean()

        # Loss
        ce_weight,rce_weight,mae_weight,mse_weight = self.activation(self.alpha_ce),self.activation(self.alpha_rce),self.activation(self.alpha_mae),self.activation(self.alpha_mse)
        loss = ce_weight * ce + rce_weight * rce + mae_weight * mae
        return loss

class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, scale=1.0, gamma=0, num_classes=10, alpha=None, size_average=True):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = self.scale * loss / normalizor

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class ReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce.mean()


class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return self.scale * nce.mean()

class NFLandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandRCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(scale=alpha, gamma=gamma, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.rce(pred, labels)


class NCEandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            module.bias.data.zero_()

class ML_3Layer_Loss(torch.nn.Module):
    def __init__(self,in_dim,hidden_dim):
        super(ML_3Layer_Loss,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim,hidden_dim[0], bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim[0],hidden_dim[1],bias=False),
            nn.ReLU(),
        )
        self.loss = nn.Sequential(nn.Linear(hidden_dim[1],1,bias=False), nn.Softplus())
        self.reset()
        self.to('cuda')

    def forward(self,y_in,y_target):
        self.num_classes = y_in.size(1)
        y_target = torch.nn.functional.one_hot(y_target, self.num_classes).float().to(device)
        y = torch.cat((y_in, y_target),dim=1)
        yp = self.layers(y)
        return self.loss(yp).mean()

    def reset(self):
        self.layers.apply(weight_init)
        self.loss.apply(weight_init)