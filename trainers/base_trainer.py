import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import os
import loss_functions

class BaseTrainer:
    def __init__(self,model,datasets,optimizer,scheduler, val_criterion,logger,result_saved_path,args):
        self.train_dataset, self.val_dataset, self.train_Cval_dataset, self.train_Nval_dataset,self.test_dataset = datasets
        self.model = model
        self.optimizer = optimizer
        self.val_criterion = val_criterion
        self.logger = logger
        self.args = args
        self.result_saved_path = result_saved_path
        self.best_val = 0
        self.best_test = 0
        self.scheduler = scheduler

        if len(args.train_loss.split('+')) == 1:
            self.train_criterion = getattr(loss_functions,args.train_loss.split('+')[0])
        elif len(args.train_loss.split('+')) > 1:
            self.train_criterions = [getattr(loss_functions, i)  for i in args.train_loss.split('+')]
        else:
            print('XXXXXXXXXXXXXX len(args.train_loss) < 1 XXXXXXXXXXXXXXXXXX')

    def adjust_learning_rate(self, epoch):
        if epoch in self.args.lr_schedule:
            self.args.lr *= 0.1
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1


    def _save_checkpoint(self,epoch,results):
        self.best_test = max(results['test_acc_1'],self.best_test)
        state = {'epoch':epoch,
                'state_dict':self.model.state_dict(),
                'acc':results['test_acc_1'],
                'best_acc':self.best_test}
        # torch.save(state,os.path.join(self.result_saved_path,'checkpoint_epoch_'+str(epoch)+'.ckp'))
        if self.best_test == results['test_acc_1']:
            torch.save(state,os.path.join(self.result_saved_path,'best_test_acc_'+str(results['test_acc_1'])+'_epoch'+str(epoch)+'.ckp'))

    

