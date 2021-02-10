import torchvision
import torch
import numpy as np
from .base_trainer import BaseTrainer
import torch.utils.data as data
from tqdm import tqdm
import os
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import pdb
import model.model as model_zoo


class Neg_Trainer(BaseTrainer):
    def __init__(self,model,datasets,logger,resuls_saved_path,args):
        super().__init__(model,datasets,logger,resuls_saved_path,args)
        self.minuend_model = getattr(model_zoo,args.model_dict['type'])(**args.model_dict['args'])
        self.minuend_model = self.minuend_model.cuda()
        self.minuend_model = self.minuend_model.load_state_dict(torch.load(args.minuend_path)['state_dict'])
        self.softmax = torch.nn.Softmax(dim=1)


    def _train_epoch(self):
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
            # print(batch_idx)
            inputs, noisy_labels, soft_labels, gt_labels = inputs.cuda(),noisy_labels.cuda(),soft_labels.cuda(),gt_labels.cuda()

            corrupted_flag = ~ noisy_labels.eq(gt_labels)

            outputs = self.model(inputs)

            loss = (self.train_criterion(outputs,noisy_labels) * corrupted_flag).mean()
            # pdb.set_trace()

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

        val_loss, val_acc1, val_acc1_minuend = self._val_epoch()
        test_loss, test_acc1, test_acc1_minuend = self._test_epoch()

        print('val_minuend_acc1:'+str(val_acc1_minuend))
        print('test_minuend_acc1:'+str(test_acc1_minuend))
        log = {'train_loss':losses.avg,
            'train_N_acc_1':Ntop1.avg,
            'train_C_acc_1':Ctop1.avg,
            'val_loss':val_loss,
            'val_acc_1':val_acc1,
            'test_loss':test_loss,
            'test_acc_1':test_acc1}
        return log


    # def _test_minuend_model(self):
    #     self.minuend_model.eval()

    #     losses = AverageMeter()
    #     top1 = AverageMeter()
    #     top5 = AverageMeter()

    #     with torch.no_grad():
    #         # with tqdm(self.test_loader) as progress:
    #         for batch_idx, (inputs,gt_labels) in enumerate(self.test_loader):
    #             inputs,gt_labels = inputs.cuda(),gt_labels.cuda()

    #             outputs = self.minuend_model(inputs)
    #             loss = self.val_criterion(outputs,gt_labels)

    #             prec1, prec5 = accuracy(outputs,gt_labels,topk=(1,5))
    #             losses.update(loss.item(),inputs.size(0))
    #             top1.update(prec1,inputs.size(0))
    #             top5.update(prec5,inputs.size(0))

    #     return losses.avg, top1.avg, top5.avg


    def _val_epoch(self):
        self.model.eval()
        self.minuend_model.eval()
        losses = AverageMeter()
        top1_final = AverageMeter()
        top1_minuend = AverageMeter()
        top5_final = AverageMeter()

        with torch.no_grad():
            # with tqdm(self.test_loader) as progress:
            for batch_idx, (inputs, noisy_labels, soft_labels, gt_labels, index) in enumerate(self.val_loader):
                inputs,gt_labels = inputs.cuda(),gt_labels.cuda()

                outputs_minuend = self.minuend_model(inputs)
                outputs = self.model(inputs)
                final_outputs = self.softmax(outputs_minuend) - self.softmax(outputs)
                loss = self.val_criterion(outputs,gt_labels)

                minuend_prec1, _ = accuracy(outputs_minuend,gt_labels,topk=(1,5))
                final_prec1, final_prec5 = accuracy(final_outputs,gt_labels,topk=(1,5))
                losses.update(loss.item(),inputs.size(0))
                top1_final.update(final_prec1,inputs.size(0))
                top5_final.update(final_prec5,inputs.size(0))
                top1_minuend.update(minuend_prec1,inputs.size(0))

        return losses.avg, top1_final.avg, top1_minuend.avg


    def _test_epoch(self):
        self.model.eval()
        self.minuend_model.eval()


        losses = AverageMeter()
        top1_final = AverageMeter()
        top1_minuend = AverageMeter()
        top5_final = AverageMeter()

        with torch.no_grad():
            # with tqdm(self.test_loader) as progress:
            for batch_idx, (inputs,gt_labels) in enumerate(self.test_loader):
                inputs,gt_labels = inputs.cuda(),gt_labels.cuda()

                outputs_minuend = self.minuend_model(inputs)
                outputs = self.model(inputs)
                final_outputs = self.softmax(outputs_minuend) - self.softmax(outputs)
                loss = self.val_criterion(outputs,gt_labels)

                minuend_prec1, _ = accuracy(outputs_minuend,gt_labels,topk=(1,5))
                final_prec1, final_prec5 = accuracy(final_outputs,gt_labels,topk=(1,5))
                losses.update(loss.item(),inputs.size(0))
                top1_final.update(final_prec1,inputs.size(0))
                top5_final.update(final_prec5,inputs.size(0))
                top1_minuend.update(minuend_prec1,inputs.size(0))

        return losses.avg, top1_final.avg, top1_minuend.avg
    
