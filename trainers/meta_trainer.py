import torchvision
import torch
import numpy as np
from .base_trainer import BaseTrainer
import torch.utils.data as data
from tqdm import tqdm
import torch.optim as optim
import os
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import pdb
import higher

class MetaTrainer(BaseTrainer):
    def __init__(self,model,datasets,logger,resuls_saved_path,args):
        super().__init__(model,datasets,logger,resuls_saved_path,args)
        self.train_loader = data.DataLoader(self.train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4)
        self.val_loader = data.DataLoader(self.val_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4)
        self.test_loader = data.DataLoader(self.test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=4)

        # self.meta_optimizer = getattr(optim,args.meta_optim['type'])(self.model.parameters(),**args.meta_optim['args'])
        self.meta_optimizer = getattr(optim,args.meta_optim['type'])(self.train_criterion.parameters(),**args.meta_optim['args'])
        self.meta_scheduler = getattr(optim.lr_scheduler,args.meta_lr_scheduler['type'])(self.meta_optimizer,**args.meta_lr_scheduler['args'])

    def _train_epoch(self,epoch):
        self.model.train()
        losses = AverageMeter()
        Ntop1 = AverageMeter()
        Ntop5 = AverageMeter()

        Ctop1 = AverageMeter()
        Ctop5 = AverageMeter()

        for batch_idx, (inputs, noisy_labels, soft_labels, gt_labels, index) in enumerate(self.train_loader):
            inner_inputs, inner_noisy_labels, inner_soft_labels, inner_gt_labels = inputs.cuda(),noisy_labels.cuda(),soft_labels.cuda(),gt_labels.cuda()

            with higher.innerloop_ctx(self.model,self.optimizer,copy_initial_weights=False) as (fnet,diffopt):
                
                # inner loop
                inner_outputs = fnet(inner_inputs)
                inner_loss = self.train_criterion(inner_outputs,inner_noisy_labels)
                pdb.set_trace()
                diffopt.step(inner_loss)



                # outer loop
                self.meta_optimizer.zero_grad()
                for batch_idx, (out_inputs, out_noisy_labels, out_soft_labels,out_gt_labels,out_index) in enumerate(self.val_loader):
                    out_inputs,out_noisy_labels,out_soft_labels,out_gt_labels = out_inputs.cuda(),out_noisy_labels.cuda(),out_soft_labels.cuda(),out_gt_labels.cuda()

                    out_outputs = fnet(out_inputs)
                    out_loss = self.val_criterion(out_outputs,out_gt_labels)
                    out_loss.backward()
                    if batch_idx == 0:
                        break

                self.meta_optimizer.step()
                

            # actual training
            outputs = self.model(inner_inputs)

            loss = self.train_criterion(outputs,inner_noisy_labels)
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

        
        test_loss, test_acc1, test_acc5 = self._test_epoch(epoch)

        log = {'train_loss':losses.avg,
            'train_N_acc_1':Ntop1.avg,
            'train_C_acc_1':Ctop1.avg,
            'test_loss':test_loss,
            'test_acc_1':test_acc1}
        return log
    



