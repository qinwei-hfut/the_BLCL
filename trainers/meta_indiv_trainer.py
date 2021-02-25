import torchvision
import torch
import numpy as np
from .base_trainer import BaseTrainer
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
import os
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import pdb
import higher

class MetaIndivTrainer(BaseTrainer):
    # 这个trainer的训练方式是每个样本都有一个单独可以meta学习loss function 权重；
    def __init__(self,model,datasets,logger,resuls_saved_path,args):
        super().__init__(model,datasets,logger,resuls_saved_path,args)
        # self.meta_val_loader = data.DataLoader(datasets[self.args.split_dataset['valset']],batch_size=self.args.meta_batch_size,shuffle=True,num_workers=4)
        # self.meta_loader = data.DataLoader(datasets[self.args.split_dataset['metaset']],batch_size=self.args.batch_size,shuffle=True,num_workers=4)

        # if self.args.extra == 'only_function':
        # self.meta_optimizer = getattr(optim,self.args.meta_optim['type'])(self.train_criterion.parameters(),**args.meta_optim['args'])
        self.meta_lr = args.meta_optim['args']['lr']
        # else:
        #     self.meta_optimizer = getattr(optim,self.args.meta_optim['type'])(self.parameters(),**args.meta_optim['args'])
        # pdb.set_trace()
        # self.meta_scheduler = getattr(optim.lr_scheduler,self.args.meta_lr_scheduler['type'])(self.meta_optimizer,**args.meta_lr_scheduler['args'])
        if self.train_criterion_dict['type'] == 'Mixed_loss':
            self.activation = getattr(torch.nn,self.train_criterion_dict['args']['activation_type'])()

        pdb.set_trace()
        self.num_classes = self.args.arch['args']['num_classes']

    def _plot_loss_weight(self):
        # pdb.set_trace()
        self.tensorplot.add_scalers('loss_weight',{
                'ce_weight':self.activation(self.train_criterion.alpha_ce).item(),
                'rce_weight':self.activation(self.train_criterion.alpha_rce).item(),
                'mae_weight':self.activation(self.train_criterion.alpha_mae).item(),
                'mse_weight':self.activation(self.train_criterion.alpha_mse).item()
            },self.epoch)
        self.writer.add_scalars('loss_weight',{
                'ce_weight':self.activation(self.train_criterion.alpha_ce).item(),
                'rce_weight':self.activation(self.train_criterion.alpha_rce).item(),
                'mae_weight':self.activation(self.train_criterion.alpha_mae).item(),
                'mse_weight':self.activation(self.train_criterion.alpha_mse).item()
            },self.epoch)


    def _train_epoch(self):
        self.model.train()
        losses = AverageMeter()
        Ntop1 = AverageMeter()
        Ntop5 = AverageMeter()

        Ctop1 = AverageMeter()
        Ctop5 = AverageMeter()

        # pdb.set_trace()
        self.train_loader = data.DataLoader(self.trainset,batch_size=self.args.batch_size,shuffle=True,num_workers=4)

        if self.train_criterion_dict['type'] == 'Mixed_loss':
            print('ce_weight:'+str(self.activation(self.train_criterion.alpha_ce).item()))
            print('rce_weight:'+str(self.activation(self.train_criterion.alpha_rce).item()))
            print('mae_weight:'+str(self.activation(self.train_criterion.alpha_mae).item()))
            print('mse_weight:'+str(self.activation(self.train_criterion.alpha_mse).item()))


        for batch_idx, (inputs, noisy_labels, extra_data, gt_labels, index) in enumerate(self.train_loader):
            inner_inputs, inner_noisy_labels, inner_extra_data, inner_gt_labels = inputs.cuda(),noisy_labels.cuda(),extra_data.cuda(),gt_labels.cuda()
            # inner_loss_weight_per_sample = inner_extra_data[:,self.num_classes:self.num_classes+4]
            inner_extra_data.requires_grad_(True)
            inner_extra_data.grad = torch.zeros(inner_extra_data.size())
            inner_extra_data.grad.zero_()

            
            with higher.innerloop_ctx(self.model,self.optimizer,copy_initial_weights=False) as (fnet,diffopt):
                
                # inner loop
                inner_outputs = fnet(inner_inputs)
                inner_loss = self.train_criterion(inner_outputs,inner_noisy_labels,inner_extra_data[:,self.num_classes:self.num_classes+4])
                diffopt.step(inner_loss)



                # outer loop
                # self.meta_optimizer.zero_grad()
                for out_batch_idx, (out_inputs, out_noisy_labels, out_soft_labels,out_gt_labels,out_index) in enumerate(self.meta_loader):
                    out_inputs,out_noisy_labels,out_soft_labels,out_gt_labels = out_inputs.cuda(),out_noisy_labels.cuda(),out_soft_labels.cuda(),out_gt_labels.cuda()
                    # print(out_index)
                    out_outputs = fnet(out_inputs)
                    out_loss = self.val_criterion(out_outputs,out_gt_labels)
                    out_loss.backward()
                    if out_batch_idx == 0:
                        break

                # self.meta_optimizer.step()
                inner_extra_data = inner_extra_data - self.args.meta_lr * inner_extra_data.grad
                self.trainset.update_extra_data(index,inner_extra_data)
                

            # actual training
            outputs = self.model(inner_inputs)

            loss = self.train_criterion(outputs,inner_noisy_labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            Nprec1, Nprec5 = accuracy(outputs,inner_noisy_labels,topk=(1,5))
            Cprec1, Cprec5 = accuracy(outputs,inner_gt_labels,topk=(1,5))
            losses.update(loss.item(), inputs.size(0))
            Ntop1.update(Nprec1.item(), inputs.size(0))
            Ntop5.update(Nprec5.item(), inputs.size(0))
            Ctop1.update(Cprec1.item(), inputs.size(0))
            Ctop5.update(Cprec5.item(), inputs.size(0))

        val_loss, val_acc1, val_acc5 = self._val_epoch()
        test_loss, test_acc1, test_acc5 = self._test_epoch()

        # self.meta_scheduler.step()

        print("meta lr: "+str(self.meta_lr))

        log = {'train_loss':losses.avg,
            'train_N_acc_1':Ntop1.avg,
            'train_C_acc_1':Ctop1.avg,
            'val_loss':val_loss,
            'val_acc_1':val_acc1,
            'test_loss':test_loss,
            'test_acc_1':test_acc1}
        return log
    



