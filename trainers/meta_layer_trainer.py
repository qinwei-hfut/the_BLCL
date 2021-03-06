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
import loss_functions

class MetaLayerTrainer(BaseTrainer):
    def __init__(self,model,datasets,logger,resuls_saved_path,args):
        super(MetaLayerTrainer,self).__init__(model,datasets,logger,resuls_saved_path,args)
        # self.meta_val_loader = data.DataLoader(datasets[self.args.split_dataset['valset']],batch_size=self.args.meta_batch_size,shuffle=True,num_workers=4)
        # self.meta_loader = data.DataLoader(datasets[self.args.split_dataset['metaset']],batch_size=self.args.batch_size,shuffle=True,num_workers=4)

        params_fc = self.model.linear.parameters()
        self.optimizer_fc = getattr(optim,args.optim['type'])(params_fc,**args.optim['args'])
        self.scheduler_fc = getattr(optim.lr_scheduler,args.lr_scheduler['type'])(self.optimizer_fc,**args.lr_scheduler['args'])
        self.train_criterion_fc = getattr(loss_functions,self.train_criterion_dict['type'])(**self.train_criterion_dict['args'])

        param_34 = self.add_generator(self.model.layer3.parameters(),self.model.layer4.parameters()) 
        self.optimizer_34 = getattr(optim,args.optim['type'])(param_34,**args.optim['args'])
        self.scheduler_34 = getattr(optim.lr_scheduler,args.lr_scheduler['type'])(self.optimizer_34,**args.lr_scheduler['args'])
        self.train_criterion_34 = getattr(loss_functions,self.train_criterion_dict['type'])(**self.train_criterion_dict['args'])

        param_12 = self.add_generator_4(self.model.conv1.parameters(),self.model.bn1.parameters(),self.model.layer1.parameters(),self.model.layer2.parameters())
        # param_12 = self.add_generator(self.model.layer1.parameters(),self.model.layer2.parameters())
        self.optimizer_12 = getattr(optim,args.optim['type'])(param_12,**args.optim['args'])
        self.scheduler_12 = getattr(optim.lr_scheduler,args.lr_scheduler['type'])(self.optimizer_12,**args.lr_scheduler['args'])
        self.train_criterion_12 = getattr(loss_functions,self.train_criterion_dict['type'])(**self.train_criterion_dict['args'])

        if self.args.extra == 'only_function':
            weight_loss_functions = self.add_generator_3(self.train_criterion_12.parameters(),self.train_criterion_34.parameters(),self.train_criterion_fc.parameters())
            self.meta_optimizer = getattr(optim,self.args.meta_optim['type'])(weight_loss_functions,**args.meta_optim['args'])
        else:
            self.meta_optimizer = getattr(optim,self.args.meta_optim['type'])(self.parameters(),**args.meta_optim['args'])
        self.meta_scheduler = getattr(optim.lr_scheduler,self.args.meta_lr_scheduler['type'])(self.meta_optimizer,**args.meta_lr_scheduler['args'])
        self.activation = getattr(torch.nn,self.train_criterion_dict['args']['activation_type'])()

    def add_generator(self,g1,g2):
        yield from g1
        yield from g2
    
    def add_generator_3(self,g1,g2,g3):
        yield from g1
        yield from g2
        yield from g3
    
    def add_generator_4(self,g1,g2,g3,g4):
        yield from g1
        yield from g2
        yield from g3
        yield from g4

    def _plot_loss_weight(self):
        # pdb.set_trace()
        self.tensorplot.add_scalers('fc_loss_weight',{
                'ce_weight':self.activation(self.train_criterion_fc.alpha_ce).item(),
                'rce_weight':self.activation(self.train_criterion_fc.alpha_rce).item(),
                'mae_weight':self.activation(self.train_criterion_fc.alpha_mae).item(),
                'mse_weight':self.activation(self.train_criterion_fc.alpha_mse).item()
            },self.epoch)
        self.writer.add_scalars('fc_loss_weight',{
                'ce_weight':self.activation(self.train_criterion_fc.alpha_ce).item(),
                'rce_weight':self.activation(self.train_criterion_fc.alpha_rce).item(),
                'mae_weight':self.activation(self.train_criterion_fc.alpha_mae).item(),
                'mse_weight':self.activation(self.train_criterion_fc.alpha_mse).item()
            },self.epoch)

        self.tensorplot.add_scalers('34_loss_weight',{
                'ce_weight':self.activation(self.train_criterion_34.alpha_ce).item(),
                'rce_weight':self.activation(self.train_criterion_34.alpha_rce).item(),
                'mae_weight':self.activation(self.train_criterion_34.alpha_mae).item(),
                'mse_weight':self.activation(self.train_criterion_34.alpha_mse).item()
            },self.epoch)
        self.writer.add_scalars('34_loss_weight',{
                'ce_weight':self.activation(self.train_criterion_34.alpha_ce).item(),
                'rce_weight':self.activation(self.train_criterion_34.alpha_rce).item(),
                'mae_weight':self.activation(self.train_criterion_34.alpha_mae).item(),
                'mse_weight':self.activation(self.train_criterion_34.alpha_mse).item()
            },self.epoch)

        self.tensorplot.add_scalers('12_loss_weight',{
                'ce_weight':self.activation(self.train_criterion_12.alpha_ce).item(),
                'rce_weight':self.activation(self.train_criterion_12.alpha_rce).item(),
                'mae_weight':self.activation(self.train_criterion_12.alpha_mae).item(),
                'mse_weight':self.activation(self.train_criterion_12.alpha_mse).item()
            },self.epoch)
        self.writer.add_scalars('12_loss_weight',{
                'ce_weight':self.activation(self.train_criterion_12.alpha_ce).item(),
                'rce_weight':self.activation(self.train_criterion_12.alpha_rce).item(),
                'mae_weight':self.activation(self.train_criterion_12.alpha_mae).item(),
                'mse_weight':self.activation(self.train_criterion_12.alpha_mse).item()
            },self.epoch)


    def _train_epoch(self):
        self.model.train()
        losses = AverageMeter()
        Ntop1 = AverageMeter()
        Ntop5 = AverageMeter()

        Ctop1 = AverageMeter()
        Ctop5 = AverageMeter()

        # pdb.set_trace()

        print('fc_ce_weight:'+str(self.activation(self.train_criterion_fc.alpha_ce).item()))
        print('fc_rce_weight:'+str(self.activation(self.train_criterion_fc.alpha_rce).item()))
        print('fc_mae_weight:'+str(self.activation(self.train_criterion_fc.alpha_mae).item()))
        print('fc_mse_weight:'+str(self.activation(self.train_criterion_fc.alpha_mse).item()))

        print('34_ce_weight:'+str(self.activation(self.train_criterion_34.alpha_ce).item()))
        print('34_rce_weight:'+str(self.activation(self.train_criterion_34.alpha_rce).item()))
        print('34_mae_weight:'+str(self.activation(self.train_criterion_34.alpha_mae).item()))
        print('34_mse_weight:'+str(self.activation(self.train_criterion_34.alpha_mse).item()))

        print('12_ce_weight:'+str(self.activation(self.train_criterion_12.alpha_ce).item()))
        print('12_rce_weight:'+str(self.activation(self.train_criterion_12.alpha_rce).item()))
        print('12_mae_weight:'+str(self.activation(self.train_criterion_12.alpha_mae).item()))
        print('12_mse_weight:'+str(self.activation(self.train_criterion_12.alpha_mse).item()))


        for batch_idx, (inputs, noisy_labels, soft_labels, gt_labels, index) in enumerate(self.train_loader):
            inner_inputs, inner_noisy_labels, inner_soft_labels, inner_gt_labels = inputs.cuda(),noisy_labels.cuda(),soft_labels.cuda(),gt_labels.cuda()

            # print(batch_idx)

            # layer1 and layer2
            self.model.zero_grad()
            with higher.innerloop_ctx(self.model,self.optimizer_12,copy_initial_weights=False) as (fnet,diffopt):
                # inner loop
                inner_outputs = fnet(inner_inputs)
                inner_loss = self.train_criterion_12(inner_outputs,inner_noisy_labels)
                diffopt.step(inner_loss)
                # outer loop
                self.meta_optimizer.zero_grad()
                for out_batch_idx, (out_inputs, out_noisy_labels, out_soft_labels,out_gt_labels,out_index) in enumerate(self.meta_loader):
                    out_inputs,out_noisy_labels,out_soft_labels,out_gt_labels = out_inputs.cuda(),out_noisy_labels.cuda(),out_soft_labels.cuda(),out_gt_labels.cuda()
                    # print(out_index)
                    out_outputs = fnet(out_inputs)
                    out_loss = self.val_criterion(out_outputs,out_gt_labels)
                    out_loss.backward()
                    if out_batch_idx == 0:
                        break
                self.meta_optimizer.step()
            # actual training
            outputs = self.model(inner_inputs)
            loss = self.train_criterion_12(outputs,inner_noisy_labels)
            self.optimizer_12.zero_grad()
            loss.backward()
            self.optimizer_12.step()


            # layer3 and layer4
            self.model.zero_grad()
            with higher.innerloop_ctx(self.model,self.optimizer_34,copy_initial_weights=False) as (fnet,diffopt):
                # inner loop
                inner_outputs = fnet(inner_inputs)
                inner_loss = self.train_criterion_34(inner_outputs,inner_noisy_labels)
                diffopt.step(inner_loss)
                # outer loop
                self.meta_optimizer.zero_grad()
                for out_batch_idx, (out_inputs, out_noisy_labels, out_soft_labels,out_gt_labels,out_index) in enumerate(self.meta_loader):
                    out_inputs,out_noisy_labels,out_soft_labels,out_gt_labels = out_inputs.cuda(),out_noisy_labels.cuda(),out_soft_labels.cuda(),out_gt_labels.cuda()
                    # print(out_index)
                    out_outputs = fnet(out_inputs)
                    out_loss = self.val_criterion(out_outputs,out_gt_labels)
                    out_loss.backward()
                    if out_batch_idx == 0:
                        break
                self.meta_optimizer.step()
            # actual training
            outputs = self.model(inner_inputs)
            loss = self.train_criterion_34(outputs,inner_noisy_labels)
            self.optimizer_34.zero_grad()
            loss.backward()
            self.optimizer_34.step()

            # fc
            self.model.zero_grad()
            with higher.innerloop_ctx(self.model,self.optimizer_fc,copy_initial_weights=False) as (fnet,diffopt):
                # inner loop
                inner_outputs = fnet(inner_inputs)
                inner_loss = self.train_criterion_fc(inner_outputs,inner_noisy_labels)
                diffopt.step(inner_loss)
                # outer loop
                self.meta_optimizer.zero_grad()
                for out_batch_idx, (out_inputs, out_noisy_labels, out_soft_labels,out_gt_labels,out_index) in enumerate(self.meta_loader):
                    out_inputs,out_noisy_labels,out_soft_labels,out_gt_labels = out_inputs.cuda(),out_noisy_labels.cuda(),out_soft_labels.cuda(),out_gt_labels.cuda()
                    # print(out_index)
                    out_outputs = fnet(out_inputs)
                    out_loss = self.val_criterion(out_outputs,out_gt_labels)
                    out_loss.backward()
                    if out_batch_idx == 0:
                        break
                self.meta_optimizer.step()
            # actual training
            outputs = self.model(inner_inputs)
            loss = self.train_criterion_fc(outputs,inner_noisy_labels)
            self.optimizer_fc.zero_grad()
            loss.backward()
            self.optimizer_fc.step()

            Nprec1, Nprec5 = accuracy(outputs,inner_noisy_labels,topk=(1,5))
            Cprec1, Cprec5 = accuracy(outputs,inner_gt_labels,topk=(1,5))
            losses.update(loss.item(), inputs.size(0))
            Ntop1.update(Nprec1.item(), inputs.size(0))
            Ntop5.update(Nprec5.item(), inputs.size(0))
            Ctop1.update(Cprec1.item(), inputs.size(0))
            Ctop5.update(Cprec5.item(), inputs.size(0))

        val_loss, val_acc1, val_acc5 = self._val_epoch()
        test_loss, test_acc1, test_acc5 = self._test_epoch()

        self.meta_scheduler.step()
        self.scheduler_12.step()
        self.scheduler_34.step()
        self.scheduler_fc.step()

        print(self.optimizer_fc.param_groups[0]['lr'])
        print(self.optimizer_34.param_groups[0]['lr'])
        print(self.optimizer_12.param_groups[0]['lr'])

        log = {'train_loss':losses.avg,
            'train_N_acc_1':Ntop1.avg,
            'train_C_acc_1':Ctop1.avg,
            'val_loss':val_loss,
            'val_acc_1':val_acc1,
            'test_loss':test_loss,
            'test_acc_1':test_acc1}
        return log
    



