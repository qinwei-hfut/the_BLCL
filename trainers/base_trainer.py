from abc import abstractmethod
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import os
import loss_functions
import time
from torch.utils.tensorboard import SummaryWriter
from myUtils.tensor_plot import TensorPlot

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
        self.writer = SummaryWriter(os.path.join(self.result_saved_path,'tensorboard_plot_'+str(time.time())))
        self.tensorplot = TensorPlot(self.result_saved_path)
        self.epoch = 0

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
    
    @abstractmethod
    def _train_epoch(self, epoch):

        raise NotImplementedError

    def train(self):
        # for epoch in tqdm(range(self.args.epochs),decs='Total progress: '):
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            print('epoch: '+str(epoch))
            # self.adjust_learning_rate(epoch)
            results = self._train_epoch(epoch)
            self.scheduler.step()
            print(results)
            self.logger.append([self.optimizer.param_groups[0]['lr'], results['train_loss'], results["test_loss"], results['train_N_acc_1'], results['train_C_acc_1'], results['test_acc_1']])
            self._tensorboard(results)
            self._save_checkpoint(epoch,results)
        self.logger.close()
        self.writer.close()

    def _tensorboard(self,logdcit):
        loss_dict = {}
        acc_dict = {}
        for k,v in logdcit.items():
            if 'loss' in k:
                loss_dict[k] = v
            elif 'acc' in k:
                acc_dict[k] = v
        
        self.writer.add_scalars('loss',loss_dict,self.epoch)
        self.writer.add_scalars('acc',acc_dict,self.epoch)
        self.writer.flush()

        self.tensorplot.add_scalers('loss',loss_dict,self.epoch)
        self.tensorplot.add_scalers('acc',acc_dict,self.epoch)
        self.writer.flush()


