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
import torch.nn.functional as F


class DoubleFC_Trainer(BaseTrainer):
    def __init__(self,model,datasets,logger,resuls_saved_path,args):
        super().__init__(model,datasets,logger,resuls_saved_path,args)
        # self.minuend_model = getattr(model_zoo,args.model_dict['type'])(**args.model_dict['args'])
        # self.minuend_model = self.minuend_model.cuda()
        # # pdb.set_trace()
        # self.minuend_model.load_state_dict(torch.load(args.minuend_path)['state_dict'])
        self.softmax = torch.nn.Softmax(dim=1)

        self.best_final = 0.0
        self.best_clean = 0.0



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

            # corrupted_flag = ~ noisy_labels.eq(gt_labels)
            # clean_flag = noisy_labels.eq(gt_labels)

            output_main, output_2 = self.model(inputs)

            # 这样操作是否batch大小，或者说是梯度大小？需要检查一下 TODO
            # loss_clean = (self.train_criterion(outputs[0],noisy_labels) * clean_flag).mean()
            # loss_corrupted = (self.train_criterion(outputs[1],noisy_labels) * corrupted_flag).mean()
            # TODO .mean()?
            loss_main = self.train_criterion(output_main,gt_labels)
            # pdb.set_trace()
            # self.optimizer.zero_grad()
            # loss_main.backward()
            # self.optimizer.step()

            
            full_batch_index = torch.tensor([i for i in range(gt_labels.size(0))],device='cuda')
            output_main_copy = output_main.clone()
            output_main_copy[full_batch_index,gt_labels] = torch.tensor(float('-inf'),device='cuda')
            _,negative_label = output_main_copy.max(dim=1)
            # pdb.set_trace()
            # if self.epoch == 90:
            #     pdb.set_trace()


            loss_2 = self.train_criterion(output_2,negative_label.detach())   #TODO .mean()?
            
            # loss_2 = self.train_criterion(output_2,gt_labels)
            
            self.optimizer.zero_grad()
            loss_main.backward()
            loss_2.backward()
            self.optimizer.step()




            

            Nprec1, Nprec5 = accuracy(output_2,negative_label.detach(),topk=(1,5))
            Cprec1, Cprec5 = accuracy(output_main,gt_labels,topk=(1,5))
            losses.update(loss_main.item(), inputs.size(0))
            Ntop1.update(Nprec1.item(), inputs.size(0))
            Ntop5.update(Nprec5.item(), inputs.size(0))
            Ctop1.update(Cprec1.item(), inputs.size(0))
            Ctop5.update(Cprec5.item(), inputs.size(0))

        val_loss, val_final_acc, val_clean_acc, val_neg_acc = self._val_epoch()
        test_loss, test_final_acc,test_clean_acc,test_neg_acc = self._test_epoch()


        log = {'train_loss':losses.avg,
            'Neg_train_acc':Ntop1.avg,
            'Pos_train_acc':Ctop1.avg,
            'val_loss':val_loss,
            'val_final_acc':val_final_acc,
            'val_clean_acc':val_clean_acc,
            'val_neg_acc':val_neg_acc,
            'test_loss':test_loss,
            'test_final_acc':test_final_acc,
            'test_clean_acc':test_clean_acc,
            'test_neg_acc':test_neg_acc,}
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

    def normalize_logit(self, logits):
        if self.args.norm_mode == 'l2':
            return F.normalize(logits[0],p=2,dim=1),F.normalize(logits[1],p=2,dim=1),
        elif self.args.norm_mode == 'softmax':
            return self.softmax(logits[0]), self.softmax(logits[1])

    def _val_epoch(self):
        self.model.eval()

        losses = AverageMeter()
        top1_final = AverageMeter()
        top1_clean = AverageMeter()
        top1_neg = AverageMeter()

        with torch.no_grad():
            # with tqdm(self.test_loader) as progress:
            for batch_idx, (inputs, noisy_labels, soft_labels, gt_labels, index) in enumerate(self.val_loader):
                inputs,gt_labels = inputs.cuda(),gt_labels.cuda()

                # 在处理两个fc之间的输出时，需要normalize一下，把每一个样本都变成单位长度？sample方向的normalize
                output_main, output_2 = self.model(inputs)

                output_clean, output_corrupted = self.normalize_logit([output_main, output_2])
                output_final = output_clean - output_corrupted

                # output_final = outputs[0]

                loss = self.val_criterion(output_final,gt_labels)

                prec1_final,_ = accuracy(output_final,gt_labels,topk=(1,5))
                prec1_clean,_ = accuracy(output_clean,gt_labels,topk=(1,5))
                prec1_neg,_ = accuracy(output_corrupted,gt_labels,topk=(1,5))
                losses.update(loss.item(),inputs.size(0))
                top1_final.update(prec1_final,inputs.size(0))
                top1_clean.update(prec1_clean,inputs.size(0))
                top1_neg.update(prec1_neg, inputs.size(0))

        return losses.avg, top1_final.avg, top1_clean.avg, top1_neg.avg


    def _test_epoch(self):
        self.model.eval()

        losses = AverageMeter()
        top1_final = AverageMeter()
        top1_clean = AverageMeter()
        top1_neg = AverageMeter()

        with torch.no_grad():
            # with tqdm(self.test_loader) as progress:
            for batch_idx, (inputs,gt_labels) in enumerate(self.test_loader):
                inputs,gt_labels = inputs.cuda(),gt_labels.cuda()

                # 在处理两个fc之间的输出时，需要normalize一下，把每一个样本都变成单位长度？sample方向的normalize
                output_main, output_2 = self.model(inputs)

                output_clean, output_corrupted = self.normalize_logit([output_main, output_2])
                output_final = output_clean - output_corrupted

                # output_final = outputs[0]

                loss = self.val_criterion(output_final,gt_labels)

                prec1_final,_ = accuracy(output_final,gt_labels,topk=(1,5))
                prec1_clean,_ = accuracy(output_clean,gt_labels,topk=(1,5))
                prec1_neg,_ = accuracy(output_corrupted,gt_labels,topk=(1,5))
                losses.update(loss.item(),inputs.size(0))
                top1_final.update(prec1_final,inputs.size(0))
                top1_clean.update(prec1_clean,inputs.size(0))
                top1_neg.update(prec1_neg, inputs.size(0))

        return losses.avg, top1_final.avg, top1_clean.avg, top1_neg.avg


    def _save_checkpoint(self,epoch,results):
        self.best_final = max(results['test_final_acc'],self.best_final)
        self.best_clean = max(results['test_clean_acc'],self.best_clean)
        state = {'epoch':epoch,
                'state_dict':self.model.state_dict(),
                'test_final_acc':results['test_final_acc'],
                'test_clean_acc':results['test_clean_acc'],
                'best_acc':self.best_final,}
        if self.epoch % 4 ==0:
            torch.save(state,os.path.join(self.result_saved_path,'checkpoints/epoch_'+str(epoch)+'.ckp'))
        if self.best_final == results['test_final_acc']:
            torch.save(state,os.path.join(self.result_saved_path,'test_final_acc'+'.ckp'))
            torch.save(torch.zeros((1)),os.path.join(self.result_saved_path,'test_final_acc_epoch_'+str(epoch)+str('_')+str(results['test_clean_acc'])+'_'+str(results['test_final_acc'])))

        if self.best_clean == results['test_clean_acc']:
            torch.save(state,os.path.join(self.result_saved_path,'test_clean_acc'+'.ckp'))
            torch.save(torch.zeros((1)),os.path.join(self.result_saved_path,'test_clean_acc_epoch_'+str(epoch)+str('_')+str(results['test_clean_acc'])+'_'+str(results['test_final_acc'])))
    