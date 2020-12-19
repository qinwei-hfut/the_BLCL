import torchvision
import torch
import numpy as np
from .base_trainer import BaseTrainer
import torch.utils.data as data
from tqdm import tqdm
import os
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


class Trainer(BaseTrainer):
    def __init__(self,model,datasets,optimizer,criterion,logger,resuls_saved_path,args):
        super.__init__(model,datasets,optimizer,criterion,logger,resuls_saved_path,args)
        self.train_loader = data.DataLoader(self.train_Nval_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4)
        self.test_loader = data.DataLoader(self.test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=4)

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

            outputs = self.model(inputs)

            loss = self.criterion(outputs,noisy_labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            Nprec1, Nprec5 = accuracy(outputs,noisy_labels,topk=(1,5))
            Cprec1, Cprec5 = accuracy(outputs,gt_labels,topk=(1,5))
            losses.update(loss.item(), inputs.size(0))
            Ntop1.update(Nprec1.item(), inputs.size(0))
            Ntop5.update(Nprec5.item(), inputs.size(0))
            Ctop1.update(Cprec1.item(), inputs.size(0))
            Ctop5.update(Cprec5.item(), inputs.size(0))
        
        test_loss, test_acc1, test_acc5 = self._test_epoch(epoch)

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

                outputs = self.model(inputs)
                loss = self.criterion(outputs,gt_labels)

                prec1, prec5 = accuracy(outputs,gt_labels,topk=(1,5))
                losses.update(loss.item(),inputs.size(0))
                top1.update(prec1,inputs.size(0))
                top5.update(prec5,inputs.size(0))

        return losses.avg, top1.avg, top5.avg

    def adjust_learning_rate(self, epoch):
        if epoch in self.args.lr_schedule:
            self.args.lr *= 0.1
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1


    def _save_checkpoint(self,epoch,results):
        self.best_test = max(results['test_acc'],self.best_test)
        state = {'epoch':epoch,
                'state_dict':self.model.state_dict(),
                'acc':results['test_acc'],
                'best_acc':self.best_test}
        torch.save(state,os.path.join(self.result_saved_path,'checkpoint_epoch_'+str(epoch)+'.ckp'))
        if self.best_test == results['test_acc']:
            torch.save(state,os.path.join(self.result_saved_path,'best_test_acc_'+results['test_acc']+'_epoch'+str(epoch)+'.ckp'))


    def train(self):
        # for epoch in tqdm(range(self.args.epochs),decs='Total progress: '):
        for epoch in range(self.args.epochs):
            self.adjust_learning_rate(epoch)
            results = self._train_epoch(epoch)
            self.logger.append(self.args.lr, results['train_loss'], results["test_loss"], results['train_N_acc_1'], results['train_C_acc_1'], results['test_acc_1'])

            self._save_checkpoint(epoch,results)
        self.logger.close()
