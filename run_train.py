import os
import pdb
import json

def run_exp(trainer='trainer',arch='PreActResNet18',batch_size=128, lr=0.1, dataset='cifar100',noise_type='sym',noise_rate=0.0,gpu=0,\
        train_criterion='CE_loss',val_criterion='CE_loss',weight_decay=1e-4,lr_schedule='40_80',\
        epochs=120):
    the_cammand = 'python train.py' \
        +' --trainer='+trainer \
        +' --arch='+arch \
        +' --batch-size='+str(batch_size) \
        +' --lr='+str(lr) \
        +' --noise-type='+noise_type \
        +' --noise-rate='+str(noise_rate)\
        +' --gpu='+str(gpu) \
        +' --dataset='+dataset \
        +' --train-loss='+train_criterion \
        +' --val-loss='+val_criterion \
        +' --weight-decay='+str(weight_decay) \
        +' --lr-schedule='+lr_schedule \
        +' --epochs='+str(epochs) \

    print(the_cammand)
    os.system(the_cammand)


gpu=0




val_criterion = '\'{"type":"CE_loss","args":{}}\''
# train_criterion = '\'{"type":"MAE_one_hot_loss","args":{}}\''
train_criterion = '\'{"type":"CE_loss","args":{}}\''
# train_criterion = '\'{"type":"SCE_loss","args":{"alpha":0.1,"beta":1.0}}\''
train_criterion = '\'{"type":"CE_MAE_loss","args":{"alpha":0.1,"beta":1.0}}\''

arch = '\'{"type":"PreActResNet18","args":{"num_classes":10}}\''
        


run_exp(trainer='trainer',arch=arch,batch_size=128, lr=0.1,noise_type='sym',noise_rate=0.4,epochs=250,lr_schedule='120_200', dataset='cifar10', train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu)
run_exp(trainer='trainer',arch=arch,batch_size=128, lr=0.1,noise_type='sym',noise_rate=0.4,epochs=250,lr_schedule='120_200', dataset='cifar10', train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu)

run_exp(trainer='trainer',arch=arch,batch_size=128, lr=0.1,noise_type='sym',noise_rate=0.8,epochs=250,lr_schedule='120_200', dataset='cifar10', train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu)
run_exp(trainer='trainer',arch=arch,batch_size=128, lr=0.1,noise_type='sym',noise_rate=0.8,epochs=250,lr_schedule='120_200', dataset='cifar10', train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu)

run_exp(trainer='trainer',arch=arch,batch_size=128, lr=0.1,noise_type='asym',noise_rate=0.4,epochs=250,lr_schedule='120_200', dataset='cifar10', train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu)
run_exp(trainer='trainer',arch=arch,batch_size=128, lr=0.1,noise_type='asym',noise_rate=0.4,epochs=250,lr_schedule='120_200', dataset='cifar10', train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu)


