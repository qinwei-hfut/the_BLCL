import os
import pdb
import json

def run_exp(trainer,arch,batch_size,dataset,noise_type,noise_rate,gpu,optim,meta_optim,\
        warm_up_criterion,train_criterion,val_criterion,lr_scheduler,meta_lr_scheduler,\
        epochs,warm_up_epochs,finetune_epochs,split_dataset,finetune_optim,finetune_lr_scheduler,\
        finetune_criterion,extra,meta_batch_size):
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
        +' --finetune-loss='+ finetune_criterion \
        +' --warm-up-epochs='+str(warm_up_epochs) \
        +' --lr-scheduler='+lr_scheduler \
        +' --meta-lr-scheduler='+meta_lr_scheduler \
        +' --epochs='+str(epochs) \
        +' --finetune-epochs='+str(finetune_epochs)\
        +' --split-dataset='+split_dataset \
        +' --finetune-optim='+finetune_optim\
        +' --finetune-lr-scheduler='+finetune_lr_scheduler\
        +' --extra='+extra \
        +' --meta-batch-size='+str(meta_batch_size) \

    print(the_cammand)
    os.system(the_cammand)


gpu=1




val_criterion = '\'{"type":"CE_loss","args":{}}\''
warm_up_criterion = '\'{"type":"CE_loss","args":{}}\''
finetune_criterion = '\'{"type":"CE_loss","args":{}}\''

# train_criterion = '\'{"type":"MAE_one_hot_loss","args":{}}\''
# train_criterion = '\'{"type":"MSE_one_hot_loss","args":{}}\''
train_criterion = '\'{"type":"CE_loss","args":{}}\''
# train_criterion = '\'{"type":"SCE_loss","args":{"alpha":0.1,"beta":1.0}}\''
# train_criterion = '\'{"type":"CE_MAE_loss","args":{"alpha":0.1,"beta":1.0}}\''
# train_criterion = '\'{"type":"CE_LS_loss","args":{}}\''

# train_criterion = '\'{"type":"Mixed_loss","args":{"alpha_ce":-2.0,"alpha_rce":3.0,"alpha_mae":-2.0,"alpha_mse":-2.0,"activation_type":"Sigmoid"}}\''
# train_criterion = '\'{"type":"Mixed_loss","args":{"alpha_ce":0.0,"alpha_rce":0.0,"alpha_mae":0.0,"alpha_mse":0.0,"activation_type":"Sigmoid"}}\''

# train_criterion = '\'{"type":"NFLandRCE","args":{"alpha":1.0,"beta":1.0,"num_classes":10}}\''
# train_criterion = '\'{"type":"NCEandRCE","args":{"alpha":1.0,"beta":1.0,"num_classes":10}}\''

arch = '\'{"type":"PreActResNet18","args":{"num_classes":10}}\''
# arch = '\'{"type":"toymodel","args":{}}\''

optim = '\'{"type":"SGD","args":{"lr":0.1,"momentum":0.9,"weight_decay":1e-4}}\''
lr_scheduler = '\'{"type":"MultiStepLR","args":{"milestones":[40,80],"gamma":0.1}}\''

meta_optim = '\'{"type":"SGD","args":{"lr":0.1,"momentum":0.9,"weight_decay":1e-4}}\''
meta_lr_scheduler = '\'{"type":"MultiStepLR","args":{"milestones":[40,80],"gamma":0.1}}\''

finetune_optim = '\'{"type":"SGD","args":{"lr":0.01,"momentum":0.9,"weight_decay":1e-4}}\''
finetune_lr_scheduler = '\'{"type":"MultiStepLR","args":{"milestones":[5],"gamma":0.1}}\''

split_dataset = '\'{"trainset":"train_Cval_dataset","valset":"val_dataset","metaset":"meta_dataset","testset":"test_set"}\''
extra= ''
dataset = '\'{"type":"CIFAR10","args":{"meta":1000,"val":1000}}\''
meta_batch_size = 128
arch = '\'{"type":"PreActResNet18","args":{"num_classes":10}}\''

        


# run_exp(trainer='meta_trainer',noise_type='sym',noise_rate=0.4,epochs=120,warm_up_epochs=5,finetune_epochs=10,dataset=dataset,batch_size=128,meta_batch_size=meta_batch_size,arch=arch,optim=optim,meta_optim=meta_optim,lr_scheduler=lr_scheduler,meta_lr_scheduler=meta_lr_scheduler,warm_up_criterion=warm_up_criterion,split_dataset=split_dataset,train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu,finetune_optim=finetune_optim,finetune_lr_scheduler=finetune_lr_scheduler,finetune_criterion=finetune_criterion,extra=extra)


run_exp(trainer='trainer',noise_type='sym',noise_rate=0.0,epochs=120,warm_up_epochs=0,finetune_epochs=0,dataset=dataset,meta_batch_size=meta_batch_size,batch_size=128,arch=arch,optim=optim,meta_optim=meta_optim,lr_scheduler=lr_scheduler,meta_lr_scheduler=meta_lr_scheduler,warm_up_criterion=warm_up_criterion,split_dataset=split_dataset,train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu,finetune_optim=finetune_optim,finetune_lr_scheduler=finetune_lr_scheduler,finetune_criterion=finetune_criterion,extra=extra)

# run_exp(trainer='meta_trainer',noise_type='sym',noise_rate=0.4,epochs=120,warm_up_epochs=5,finetune_epochs=10,dataset=dataset,meta_batch_size=meta_batch_size,batch_size=128,arch=arch,optim=optim,meta_optim=meta_optim,lr_scheduler=lr_scheduler,meta_lr_scheduler=meta_lr_scheduler,warm_up_criterion=warm_up_criterion,split_dataset=split_dataset,train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu,finetune_optim=finetune_optim,finetune_lr_scheduler=finetune_lr_scheduler,finetune_criterion=finetune_criterion,extra=extra)
# run_exp(trainer='meta_trainer',noise_type='sym',noise_rate=0.8,epochs=120,warm_up_epochs=5,finetune_epochs=10,dataset=dataset,meta_batch_size=meta_batch_size,batch_size=128,arch=arch,optim=optim,meta_optim=meta_optim,lr_scheduler=lr_scheduler,meta_lr_scheduler=meta_lr_scheduler,warm_up_criterion=warm_up_criterion,split_dataset=split_dataset,train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu,finetune_optim=finetune_optim,finetune_lr_scheduler=finetune_lr_scheduler,finetune_criterion=finetune_criterion,extra=extra)
# run_exp(trainer='meta_trainer',noise_type='sym',noise_rate=0.4,epochs=120,warm_up_epochs=5,finetune_epochs=10,dataset=dataset,meta_batch_size=meta_batch_size,batch_size=128,arch=arch,optim=optim,meta_optim=meta_optim,lr_scheduler=lr_scheduler,meta_lr_scheduler=meta_lr_scheduler,warm_up_criterion=warm_up_criterion,split_dataset=split_dataset,train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu,finetune_optim=finetune_optim,finetune_lr_scheduler=finetune_lr_scheduler,finetune_criterion=finetune_criterion,extra=extra)
# run_exp(trainer='meta_trainer',noise_type='sym',noise_rate=0.8,epochs=120,warm_up_epochs=5,finetune_epochs=10,dataset=dataset,meta_batch_size=meta_batch_size,batch_size=128,arch=arch,optim=optim,meta_optim=meta_optim,lr_scheduler=lr_scheduler,meta_lr_scheduler=meta_lr_scheduler,warm_up_criterion=warm_up_criterion,split_dataset=split_dataset,train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu,finetune_optim=finetune_optim,finetune_lr_scheduler=finetune_lr_scheduler,finetune_criterion=finetune_criterion,extra=extra)

# run_exp(trainer='meta_trainer',noise_type='asym',noise_rate=0.4,epochs=120,warm_up_epochs=5,finetune_epochs=10,dataset=dataset,meta_batch_size=meta_batch_size,batch_size=128,arch=arch,optim=optim,meta_optim=meta_optim,lr_scheduler=lr_scheduler,meta_lr_scheduler=meta_lr_scheduler,warm_up_criterion=warm_up_criterion,split_dataset=split_dataset,train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu,finetune_optim=finetune_optim,finetune_lr_scheduler=finetune_lr_scheduler,finetune_criterion=finetune_criterion,extra=extra)
# run_exp(trainer='meta_trainer',noise_type='asym',noise_rate=0.4,epochs=120,warm_up_epochs=5,finetune_epochs=10,dataset=dataset,meta_batch_size=meta_batch_size,batch_size=128,arch=arch,optim=optim,meta_optim=meta_optim,lr_scheduler=lr_scheduler,meta_lr_scheduler=meta_lr_scheduler,warm_up_criterion=warm_up_criterion,split_dataset=split_dataset,train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu,finetune_optim=finetune_optim,finetune_lr_scheduler=finetune_lr_scheduler,finetune_criterion=finetune_criterion,extra=extra)
# run_exp(trainer='meta_trainer',noise_type='asym',noise_rate=0.2,epochs=120,warm_up_epochs=5,finetune_epochs=10,dataset=dataset,meta_batch_size=meta_batch_size,batch_size=128,arch=arch,optim=optim,meta_optim=meta_optim,lr_scheduler=lr_scheduler,meta_lr_scheduler=meta_lr_scheduler,warm_up_criterion=warm_up_criterion,split_dataset=split_dataset,train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu,finetune_optim=finetune_optim,finetune_lr_scheduler=finetune_lr_scheduler,finetune_criterion=finetune_criterion,extra=extra)
# run_exp(trainer='meta_trainer',noise_type='asym',noise_rate=0.2,epochs=120,warm_up_epochs=5,finetune_epochs=10,dataset=dataset,meta_batch_size=meta_batch_size,batch_size=128,arch=arch,optim=optim,meta_optim=meta_optim,lr_scheduler=lr_scheduler,meta_lr_scheduler=meta_lr_scheduler,warm_up_criterion=warm_up_criterion,split_dataset=split_dataset,train_criterion=train_criterion,val_criterion=val_criterion,gpu=gpu,finetune_optim=finetune_optim,finetune_lr_scheduler=finetune_lr_scheduler,finetune_criterion=finetune_criterion,extra=extra)


