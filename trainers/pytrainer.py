import torchvision
import torch
import numpy as np
from .base_trainer import BaseTrainer
import torch.utils.data as data
from tqdm import tqdm
import os
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import pdb
import loss_functions


class PyTrainer(BaseTrainer):
    def __init__(self,model,datasets,optimizer,scheduler,logger,resuls_saved_path,args):
        super().__init__(model,datasets,optimizer,scheduler,logger,resuls_saved_path,args)


    def _train_epoch(self,epoch):
        self.model.train()
        losses = AverageMeter()
        Ntop1 = AverageMeter()
        Ntop5 = AverageMeter()

        Ctop1 = AverageMeter()
        Ctop5 = AverageMeter()

        # with tqdm(self.train_loader) as progress:
        #     for batch_idx, (inputs, noisy_labels, soft_labels, gt_labels, index) in enumerate(progress):
        for batch_idx, (inputs, noisy_labels, soft_labels, gt_labels, index) in enumerate(self.train_loader):
            # progress.set_description_str(f'Train epoch {epoch}')
            inputs, noisy_labels, soft_labels, gt_labels = inputs.cuda(),noisy_labels.cuda(),soft_labels.cuda(),gt_labels.cuda()

            outputs,outputs_3,outputs_2,outputs_1, = self.model(inputs)
            loss_4 = self.train_criterions[0](outputs,noisy_labels)
            loss_2 = self.train_criterions[1](outputs_2,noisy_labels)

            loss = loss_4+loss_2

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

        
        test_loss, test_acc1, test_acc5 = self._test_epoch()

        log = {'train_loss':losses.avg,
            'train_N_acc_1':Ntop1.avg,
            'train_C_acc_1':Ctop1.avg,
            'test_loss':test_loss,
            'test_acc_1':test_acc1}
        return log
    
    def _test_epoch(self,epoch):
        self.model.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.no_grad():
            # with tqdm(self.test_loader) as progress:
            for index, (inputs,gt_labels) in enumerate(self.test_loader):
                inputs,gt_labels = inputs.cuda(),gt_labels.cuda()

                outputs = self.model(inputs)[0]
                loss = self.val_criterion(outputs,gt_labels)

                prec1, prec5 = accuracy(outputs,gt_labels,topk=(1,5))
                losses.update(loss.item(),inputs.size(0))
                top1.update(prec1,inputs.size(0))
                top5.update(prec5,inputs.size(0))

        return losses.avg, top1.avg, top5.avg




    # def train(self):
    #     # for epoch in tqdm(range(self.args.epochs),decs='Total progress: '):
    #     for epoch in range(self.args.epochs):
    #         print('epoch: '+str(epoch))
    #         # self.adjust_learning_rate(epoch)
    #         results = self._train_epoch(epoch)
    #         self.scheduler.step()
    #         print(results)
    #         self.logger.append([self.optimizer.param_groups[0]['lr'], results['train_loss'], results["test_loss"], results['train_N_acc_1'], results['train_C_acc_1'], results['test_acc_1']])

    #         self._save_checkpoint(epoch,results)
    #     self.logger.close()
