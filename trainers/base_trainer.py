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
import torch.optim as optim
import json
import pdb
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

class BaseTrainer(torch.nn.Module):
    def __init__(self,model,datasets,logger,result_saved_path,args):
        super(BaseTrainer,self).__init__()
        self.train_dataset, self.val_dataset, self.train_Cval_dataset, self.train_Nval_dataset,self.test_dataset = datasets
        self.model = model
        self.logger = logger
        self.args = args
        self.result_saved_path = result_saved_path
        self.best_val = 0
        self.best_test = 0
        self.writer = SummaryWriter(os.path.join(self.result_saved_path,'tensorboard_plot_'+str(time.time())))
        self.tensorplot = TensorPlot(os.path.join(self.result_saved_path,'plot'))
        self.epoch = 0

        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer = getattr(optim,args.optim['type'])(self.model.parameters(),**args.optim['args'])
        self.scheduler = getattr(optim.lr_scheduler,args.lr_scheduler['type'])(self.optimizer,**args.lr_scheduler['args'])
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=args.lr_schedule,gamma=0.1)

        val_criterion_dict = json.loads(self.args.val_loss)
        self.val_criterion = getattr(loss_functions,val_criterion_dict['type'])(**val_criterion_dict['args'])

        if len(self.args.train_loss.split('+')) == 1:
            # print(self.args.train_loss)
            train_criterion_dict = json.loads(self.args.train_loss)
            self.train_criterion = getattr(loss_functions,train_criterion_dict['type'])(**train_criterion_dict['args'])
        elif len(args.train_loss.split('+')) > 1:
            # TODO
            self.train_criterions = [getattr(loss_functions, i)  for i in args.train_loss.split('+')]
        else:
            print('XXXXXXXXXXXXXX len(args.train_loss) < 1 XXXXXXXXXXXXXXXXXX')


    # def adjust_learning_rate(self, epoch):
    #     if epoch in self.args.lr_schedule:
    #         self.args.lr *= 0.1
    #         for param_group in self.optimizer.param_groups:
    #             param_group['lr'] *= 0.1


    def _save_checkpoint(self,epoch,results):
        self.best_test = max(results['test_acc_1'],self.best_test)
        state = {'epoch':epoch,
                'state_dict':self.model.state_dict(),
                'acc':results['test_acc_1'],
                'best_acc':self.best_test}
        # torch.save(state,os.path.join(self.result_saved_path,'checkpoint_epoch_'+str(epoch)+'.ckp'))
        if self.best_test == results['test_acc_1']:
            torch.save(state,os.path.join(self.result_saved_path,'best_test_acc'+'.ckp'))
            torch.save(torch.zeros((1)),os.path.join(self.result_saved_path,'best_test_acc_'+str(results['test_acc_1'])+'_epoch'+str(epoch)))
    
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
            self._plot(results)
            self._save_checkpoint(epoch,results)
        self.logger.close()
        self.writer.close()

    def _plot(self,logdcit):
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
        self.tensorplot.flush()


    def _test_epoch(self,epoch):
        self.model.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.no_grad():
            # with tqdm(self.test_loader) as progress:
            for index, (inputs,gt_labels) in enumerate(self.test_loader):
                inputs,gt_labels = inputs.cuda(),gt_labels.cuda()

                outputs = self.model(inputs)
                loss = self.val_criterion(outputs,gt_labels)

                prec1, prec5 = accuracy(outputs,gt_labels,topk=(1,5))
                losses.update(loss.item(),inputs.size(0))
                top1.update(prec1,inputs.size(0))
                top5.update(prec5,inputs.size(0))

        return losses.avg, top1.avg, top5.avg

