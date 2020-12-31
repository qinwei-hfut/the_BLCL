import os
import pdb

def run_exp(trainer='trainer',arch='PreActResNet18',batch_size=128, lr=0.1,noise_type='sym',noise_rate=0.0,gpu=0,\
        train_criterion='ce_loss',val_criterion='ce_loss',weight_decay=1e-4,lr_schedule='40_80',\
        epochs=120):
    the_cammand = 'python train.py' \
        +' --trainer='+trainer \
        +' --arch='+arch \
        +' --batch-size='+str(batch_size) \
        +' --lr='+str(lr) \
        +' --noise-type='+noise_type \
        +' --noise-rate='+str(noise_rate)\
        +' --gpu='+str(gpu) \
        +' --train-loss='+train_criterion \
        +' --weight-decay='+str(weight_decay) \
        +' --lr-schedule='+lr_schedule \
        +' --epochs='+str(epochs) \

    print(the_cammand)
    os.system(the_cammand)


gpu=2


train_criterion = '{     \
    "type":"MAE",        \
    "args":{}      }'





run_exp(trainer='trainer',arch='PreActResNet18',batch_size=128, lr=0.1,noise_type='asym',noise_rate=0.9,gpu=gpu,lr_schedule='120_200',train_criterion=train_criterion,epochs=250)
# run_exp(trainer='trainer',arch='PreActResNet18',batch_size=128, lr=0.5,noise_type='asym',noise_rate=0.4,gpu=gpu,lr_schedule='250_400',train_criterion='MAE_loss',epochs=500)

# run_exp(trainer='trainer',arch='PreActResNet18',batch_size=128, lr=1.0,noise_type='asym',noise_rate=0.4,gpu=gpu,lr_schedule='120_200',train_criterion='MAE_loss',epochs=250)
# run_exp(trainer='trainer',arch='PreActResNet18',batch_size=128, lr=1.0,noise_type='asym',noise_rate=0.4,gpu=gpu,lr_schedule='250_400',train_criterion='MAE_loss',epochs=500)


