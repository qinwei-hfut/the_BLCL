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
import torch.utils.data as data
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

class BaseTrainer(torch.nn.Module):
    def __init__(self,model,datasets,logger,result_saved_path,args):
        super(BaseTrainer,self).__init__()
        # self.train_dataset, self.val_dataset, self.train_Cval_dataset, self.train_Nval_dataset,self.test_dataset = datasets
        self.args = args
        self.train_loader = data.DataLoader(datasets[self.args.split_dataset['trainset']],batch_size=self.args.batch_size,shuffle=True,num_workers=4)
        self.val_loader = data.DataLoader(datasets[self.args.split_dataset['valset']],batch_size=self.args.batch_size,shuffle=False,num_workers=4)
        self.test_loader = data.DataLoader(datasets[self.args.split_dataset['testset']],batch_size=self.args.batch_size,shuffle=False,num_workers=4)
        self.meta_loader = data.DataLoader(datasets[self.args.split_dataset['metaset']],batch_size=self.args.batch_size,shuffle=True,num_workers=4)
        # pdb.set_trace()
        self.model = model
        self.logger = logger
        
        self.result_saved_path = result_saved_path
        self.best_val = 0
        self.best_test = 0
        self.writer = SummaryWriter(os.path.join(self.result_saved_path,'tensorboard_plot_'+str(time.time())))
        self.tensorplot = TensorPlot(os.path.join(self.result_saved_path,'plot'))
        self.epoch = 0
        self.warm_up_epochs = self.args.warm_up_epochs

        self.optimizer = getattr(optim,args.optim['type'])(self.model.parameters(),**args.optim['args'])
        self.scheduler = getattr(optim.lr_scheduler,args.lr_scheduler['type'])(self.optimizer,**args.lr_scheduler['args'])

        self.finetune_optimizer = getattr(optim,args.finetune_optim['type'])(self.model.parameters(),**args.finetune_optim['args'])
        self.finetune_scheduler = getattr(optim.lr_scheduler,self.args.finetune_lr_scheduler['type'])(self.finetune_optimizer,**self.args.finetune_lr_scheduler['args'])

        self.val_criterion = getattr(loss_functions,self.args.val_loss['type'])(**self.args.val_loss['args'])
        self.warm_up_criterion = getattr(loss_functions,self.args.warm_up_loss['type'])(**self.args.warm_up_loss['args'])
        self.finetune_criterion = getattr(loss_functions,self.args.finetune_loss['type'])(**self.args.finetune_loss['args'])

        if len(self.args.train_loss.split('+')) == 1:
            # print(self.args.train_loss)
            self.train_criterion_dict = json.loads(self.args.train_loss)
            self.train_criterion = getattr(loss_functions,self.train_criterion_dict['type'])(**self.train_criterion_dict['args'])
        elif len(args.train_loss.split('+')) > 1:
            # TODO
            self.train_criterions = [getattr(loss_functions, i)  for i in args.train_loss.split('+')]
        else:
            print('XXXXXXXXXXXXXX len(args.train_loss) < 1 XXXXXXXXXXXXXXXXXX')

        if not os.path.exists(os.path.join(self.result_saved_path,'checkpoints')):
            os.makedirs(os.path.join(self.result_saved_path,'checkpoints'))

        self.optimizer_fc = None


    def _save_checkpoint(self,epoch,results):
        self.best_test = max(results['test_acc_1'],self.best_test)
        self.best_val = max(results['val_acc_1'],self.best_val)
        state = {'epoch':epoch,
                'state_dict':self.model.state_dict(),
                'test_acc':results['test_acc_1'],
                'val_acc':results['val_acc_1'],
                'best_acc':self.best_test,}
        if self.epoch % 4 ==0:
            torch.save(state,os.path.join(self.result_saved_path,'checkpoints/epoch_'+str(epoch)+'.ckp'))
        if self.best_test == results['test_acc_1']:
            torch.save(state,os.path.join(self.result_saved_path,'best_test_acc'+'.ckp'))
            torch.save(torch.zeros((1)),os.path.join(self.result_saved_path,'best_test_acc_epoch_'+str(epoch)+str('_')+str(results['val_acc_1'])+'_'+str(results['test_acc_1'])))

        if self.best_val == results['val_acc_1']:
            torch.save(state,os.path.join(self.result_saved_path,'best_val_acc'+'.ckp'))
            torch.save(torch.zeros((1)),os.path.join(self.result_saved_path,'best_val_acc_epoch_'+str(epoch)+str('_')+str(results['val_acc_1'])+'_'+str(results['test_acc_1'])))
    
    @abstractmethod
    def _train_epoch(self):

        raise NotImplementedError

    @abstractmethod
    def _plot_loss_weight(self):

        raise NotImplementedError

    def train(self):
        # for epoch in tqdm(range(self.args.epochs),decs='Total progress: '):
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            # if 'meta' in self.args.trainer:  
            if self.train_criterion_dict['type'] == 'Mixed_loss':   
                self._plot_loss_weight()       
            if self.epoch < self.warm_up_epochs:
                print('warm_epoch: '+str(self.epoch))
                results = self._warm_up()
                self.scheduler.step()
                self.logger.append([self.optimizer.param_groups[0]['lr'], results['train_loss'], results["val_loss"],results["test_loss"], results['train_N_acc_1'], results['train_C_acc_1'], results['val_acc_1'], results['test_acc_1']])
            elif self.epoch < (self.args.epochs - self.args.finetune_epochs):
                print('train_epoch: '+str(self.epoch))
                results = self._train_epoch()
                self.scheduler.step()
                if 'meta_layer' in self.args.trainer:
                    self.logger.append([self.optimizer_fc.param_groups[0]['lr'], results['train_loss'], results["val_loss"],results["test_loss"], results['train_N_acc_1'], results['train_C_acc_1'], results['val_acc_1'], results['test_acc_1']])
                self.logger.append([self.optimizer.param_groups[0]['lr'], results['train_loss'], results["val_loss"],results["test_loss"], results['train_N_acc_1'], results['train_C_acc_1'], results['val_acc_1'], results['test_acc_1']])
            else:
                if self.epoch == (self.args.epochs - self.args.finetune_epochs):
                    state_dict = torch.load(os.path.join(self.result_saved_path,'best_test_acc'+'.ckp'))['state_dict']
                    self.model.load_state_dict(state_dict)
                print('finetune_epoch: '+str(self.epoch))
                results = self._finetune()
                self.finetune_scheduler.step()
                self.logger.append([self.finetune_optimizer.param_groups[0]['lr'], results['train_loss'], results["val_loss"],results["test_loss"], results['train_N_acc_1'], results['train_C_acc_1'], results['val_acc_1'], results['test_acc_1']])

            
            print(results)
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

    def _val_epoch(self):
        self.model.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.no_grad():
            # with tqdm(self.test_loader) as progress:
            for batch_idx, (inputs, noisy_labels, soft_labels, gt_labels, index) in enumerate(self.val_loader):
                inputs,gt_labels = inputs.cuda(),gt_labels.cuda()

                outputs = self.model(inputs)
                loss = self.val_criterion(outputs,gt_labels)

                prec1, prec5 = accuracy(outputs,gt_labels,topk=(1,5))
                losses.update(loss.item(),inputs.size(0))
                top1.update(prec1,inputs.size(0))
                top5.update(prec5,inputs.size(0))

        return losses.avg, top1.avg, top5.avg


    def _test_epoch(self):
        self.model.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.no_grad():
            # with tqdm(self.test_loader) as progress:
            for batch_idx, (inputs,gt_labels) in enumerate(self.test_loader):
                inputs,gt_labels = inputs.cuda(),gt_labels.cuda()

                outputs = self.model(inputs)
                loss = self.val_criterion(outputs,gt_labels)

                prec1, prec5 = accuracy(outputs,gt_labels,topk=(1,5))
                losses.update(loss.item(),inputs.size(0))
                top1.update(prec1,inputs.size(0))
                top5.update(prec5,inputs.size(0))

        return losses.avg, top1.avg, top5.avg

    def _warm_up(self):
        self.model.train()
        losses = AverageMeter()
        Ntop1 = AverageMeter()
        Ntop5 = AverageMeter()

        Ctop1 = AverageMeter()
        Ctop5 = AverageMeter()

        for batch_idx, (inputs, noisy_labels, soft_labels, gt_labels, index) in enumerate(self.train_loader):

            inputs, noisy_labels, soft_labels, gt_labels = inputs.cuda(),noisy_labels.cuda(),soft_labels.cuda(),gt_labels.cuda()

            outputs = self.model(inputs)

            loss = self.warm_up_criterion(outputs,noisy_labels)

            self.optimizer.zero_grad()
            loss.backward()

            # ################ print log
            # for group in self.optimizer.param_groups:
            #     for p in group['params']:
            #         print(p.grad)


            self.optimizer.step()

            Nprec1, Nprec5 = accuracy(outputs,noisy_labels,topk=(1,5))
            Cprec1, Cprec5 = accuracy(outputs,gt_labels,topk=(1,5))
            losses.update(loss.item(), inputs.size(0))
            Ntop1.update(Nprec1.item(), inputs.size(0))
            Ntop5.update(Nprec5.item(), inputs.size(0))
            Ctop1.update(Cprec1.item(), inputs.size(0))
            Ctop5.update(Cprec5.item(), inputs.size(0))

        val_loss, val_acc1, val_acc5 = self._val_epoch()
        test_loss, test_acc1, test_acc5 = self._test_epoch()

        log = {'train_loss':losses.avg,
            'train_N_acc_1':Ntop1.avg,
            'train_C_acc_1':Ctop1.avg,
            'val_loss':val_loss,
            'val_acc_1':val_acc1,
            'test_loss':test_loss,
            'test_acc_1':test_acc1}
        return log
    
    def _finetune(self):
        self.model.train()
        losses = AverageMeter()
        Ntop1 = AverageMeter()
        Ntop5 = AverageMeter()

        Ctop1 = AverageMeter()
        Ctop5 = AverageMeter()

        for batch_idx, (inputs, noisy_labels, soft_labels, gt_labels, index) in enumerate(self.meta_loader):

            if batch_idx % 100 == 0:
                print(batch_idx)
            if batch_idx == 500:
                break

            inputs, noisy_labels, soft_labels, gt_labels = inputs.cuda(),noisy_labels.cuda(),soft_labels.cuda(),gt_labels.cuda()

            outputs = self.model(inputs)

            loss = self.finetune_criterion(outputs,gt_labels)

            self.finetune_optimizer.zero_grad()
            loss.backward()
            self.finetune_optimizer.step()

            Nprec1, Nprec5 = accuracy(outputs,noisy_labels,topk=(1,5))
            Cprec1, Cprec5 = accuracy(outputs,gt_labels,topk=(1,5))
            losses.update(loss.item(), inputs.size(0))
            Ntop1.update(Nprec1.item(), inputs.size(0))
            Ntop5.update(Nprec5.item(), inputs.size(0))
            Ctop1.update(Cprec1.item(), inputs.size(0))
            Ctop5.update(Cprec5.item(), inputs.size(0))

        val_loss, val_acc1, val_acc5 = self._val_epoch()
        test_loss, test_acc1, test_acc5 = self._test_epoch()

        log = {'train_loss':losses.avg,
            'train_N_acc_1':Ntop1.avg,
            'train_C_acc_1':Ctop1.avg,
            'val_loss':val_loss,
            'val_acc_1':val_acc1,
            'test_loss':test_loss,
            'test_acc_1':test_acc1}
        return log

