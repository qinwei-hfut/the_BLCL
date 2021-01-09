import os
import pdb
import json

def run_exp(trainer,arch,batch_size,dataset,noise_type,noise_rate,gpu,optim,meta_optim,\
        warm_up_criterion,train_criterion,val_criterion,lr_scheduler,meta_lr_scheduler,\
        epochs,warm_up_epochs,split_dataset):
    the_cammand = 'python train.py' \
        +' --trainer='+trainer \
        +' --arch='+arch \
        +' --batch-size='+str(batch_size) \
        +' --optim='+optim \
        +' --meta-optim='+meta_optim \
        +' --noise-type='+noise_type \
        +' --noise-rate='+str(noise_rate)\
        +' --gpu='+str(gpu) \
        +' --dataset='+dataset \
        +' --train-loss='+train_criterion \
        +' --val-loss='+val_criterion \
        +' --warm-up-loss='+warm_up_criterion \
        +' --warm-up-epochs='+str(warm_up_epochs) \
        +' --lr-scheduler='+lr_scheduler \
        +' --meta-lr-scheduler='+meta_lr_scheduler \
        +' --epochs='+str(epochs) \
        +' --split-dataset='+split_dataset \

    print(the_cammand)
    os.system(the_cammand)


gpu=0




val_criterion = '\'{"type":"CE_loss","args":{}}\''
warm_up_criterion = '\'{"type":"CE_loss","args":{}}\''
# train_criterion = '\'{"type":"MAE_one_hot_loss","args":{}}\''
# train_criterion = '\'{"type":"CE_loss","args":{}}\''
# train_criterion = '\'{"type":"SCE_loss","args":{"alpha":0.1,"beta":1.0}}\''
# train_criterion = '\'{"type":"CE_MAE_loss","args":{"alpha":0.1,"beta":1.0}}\''
# train_criterion = '\'{"type":"CE_LS_loss","args":{}}\''
train_criterion = '\'{"type":"Mixed_loss","args":{"alpha_ce":0.1,"alpha_rce":1.0,"alpha_mae":0.0,"alpha_mse":0.0}}\''

arch = '\'{"type":"PreActResNet18","args":{"num_classes":10}}\''

optim = '\'{"type":"SGD","args":{"lr":0.1,"momentum":0.9,"weight_decay":1e-4}}\''
lr_scheduler = '\'{"type":"MultiStepLR","args":{"milestones":[40,80],"gamma":0.1}}\''

meta_optim = '\'{"type":"SGD","args":{"lr":0.1,"momentum":0.9,"weight_decay":1e-4}}\''
meta_lr_scheduler = '\'{"type":"MultiStepLR","args":{"milestones":[40,80],"gamma":0.1}}\''
split_dataset = '\'{"trainset":"train_Cval_dataset","valset":"val_dataset","testset":"testset"}\''
extra= ''

        


run_exp(trainer='meta_trainer',arch=arch,batch_size=128,optim=optim,meta_optim=meta_optim,lr_scheduler=lr_scheduler,meta_lr_scheduler=meta_lr_scheduler,noise_type='sym',noise_rate=0.4,epochs=120,warm_up_epochs=5,dataset='cifar10',warm_up_criterion=warm_up_criterion,split_dataset=split_dataset,train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu)

# run_exp(trainer='trainer',arch=arch,batch_size=128, lr=0.1,noise_type='sym',noise_rate=0.4,epochs=250,lr_schedule='120_200', dataset='cifar10', train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu)

# run_exp(trainer='trainer',arch=arch,batch_size=128, lr=0.1,noise_type='sym',noise_rate=0.8,epochs=250,lr_schedule='120_200', dataset='cifar10', train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu)
# run_exp(trainer='trainer',arch=arch,batch_size=128, lr=0.1,noise_type='sym',noise_rate=0.8,epochs=250,lr_schedule='120_200', dataset='cifar10', train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu)

# run_exp(trainer='trainer',arch=arch,batch_size=128, lr=0.1,noise_type='asym',noise_rate=0.4,epochs=250,lr_schedule='120_200', dataset='cifar10', train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu)
# run_exp(trainer='trainer',arch=arch,batch_size=128, lr=0.1,noise_type='asym',noise_rate=0.4,epochs=250,lr_schedule='120_200', dataset='cifar10', train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu)


