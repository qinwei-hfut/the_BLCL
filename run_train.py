import os

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


gpu=0
# run_exp(arch='PreActResNet18',weight_decay=0,batch_size=256, lr=0.01,noise_type='sym',noise_rate=0.4,gpu=gpu,train_criterion='Taylor_ce_loss_1')
# run_exp(arch='PreActResNet18',batch_size=128, lr=0.1,noise_type='sym',noise_rate=0.4,gpu=gpu,train_criterion='MAE_loss')

# run_exp(arch='PyPreActResNet18',batch_size=128, lr=0.1,noise_type='sym',noise_rate=0.8,gpu=gpu,lr_schedule='120_200',train_criterion='MAE_loss',epochs=250)




run_exp(trainer='pytrainer',arch='PyPreActResNet18',batch_size=128, lr=0.1,noise_type='asym',noise_rate=0.2,gpu=gpu,lr_schedule='250_400',train_criterion='MAE_loss+MSE_loss',epochs=500)
# run_exp(trainer='trainer',arch='PreActResNet18',batch_size=128, lr=0.1,noise_type='asym',noise_rate=0.4,gpu=gpu,lr_schedule='800_1200',train_criterion='MAE_loss',epochs=1500)
# run_exp(trainer='trainer',arch='PreActResNet18',batch_size=128, lr=0.1,noise_type='asym',noise_rate=0.4,gpu=gpu,lr_schedule='1200_1600',train_criterion='MAE_loss',epochs=2000)

# run_exp(trainer='trainer',arch='PreActResNet18',batch_size=128, lr=0.1,noise_type='asym',noise_rate=0.4,gpu=gpu,lr_schedule='40_80',train_criterion='MSE_loss',epochs=140)


