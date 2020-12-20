import torch
import torchvision
import numpy as np
import torch.nn.functional as F

class BaseTrainer:
    def __init__(self,model,datasets,optimizer,train_criterion, val_criterion,logger,result_saved_path,args):
        self.train_dataset, self.val_dataset, self.train_Cval_dataset, self.train_Nval_dataset,self.test_dataset = datasets
        self.model = model
        self.optimizer = optimizer
        self.train_criterion = train_criterion
        self.val_criterion = val_criterion
        self.logger = logger
        self.args = args
        self.result_saved_path = result_saved_path
        self.best_val = 0
        self.best_test = 0

    def ce_loss(self,output,target):
        return F.cross_entropy(output,target)

    

