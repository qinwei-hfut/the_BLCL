import os

def run_exp(batch_size=128, lr=0.1,noise_type='sym',noise_rate=0.0,gpu=0):
    the_cammand = 'python train.py' \
        +' --batch-size='+str(batch_size) \
        +' --lr='+str(lr) \
        +' --noise-type='+noise_type \
        +' --noise-rate='+str(noise_rate)\
        +' --gpu='+str(gpu) \

    os.system(the_cammand)

gpu=2
run_exp(gpu=gpu)