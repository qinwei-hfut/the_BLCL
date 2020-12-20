import os

def run_exp(arch='PreActResNet18',batch_size=128, lr=0.1,noise_type='sym',noise_rate=0.0,gpu=0,\
        train_criterion='ce_loss',val_criterion='ce_loss'):
    the_cammand = 'python train.py' \
        +' --batch-size='+str(batch_size) \
        +' --lr='+str(lr) \
        +' --noise-type='+noise_type \
        +' --noise-rate='+str(noise_rate)\
        +' --gpu='+str(gpu) \
        +' --train-loss='+train_criterion \

    os.system(the_cammand)


gpu=2
run_exp(arch='PreActResNet18',batch_size=128, lr=0.1,noise_type='sym',noise_rate=0.4,gpu=gpu,train_criterion='ce_loss')
# run_exp(arch='PreActResNet18',batch_size=128, lr=0.1,noise_type='sym',noise_rate=0.4,gpu=gpu,train_criterion='MSE_loss')