from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


import trainers.trainer as trainer
import trainers.pytrainer as pytrainer
import model.model as model
import dataset.dataset as dataset
import trainers.trainers as trainers
import loss_functions
import json
import pdb


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')

# Optimization options
parser.add_argument('--trainer',type=str)
parser.add_argument('--dataset',type=json.loads)
parser.add_argument('--arch',default='PreActResNet18',type=str)
parser.add_argument('--train-loss',type=str)
parser.add_argument('--val-loss',type=json.loads)
parser.add_argument('--warm-up-loss',type=json.loads)
parser.add_argument('--finetune-loss',type=json.loads)
parser.add_argument('--epochs', default=140, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warm-up-epochs',default=5,type=int)
parser.add_argument('--finetune-epochs',type=int)
parser.add_argument('--lr-scheduler',type=json.loads)
parser.add_argument('--optim',type=json.loads)
parser.add_argument('--meta-optim',type=json.loads)
parser.add_argument('--meta-lr-scheduler',type=json.loads)
parser.add_argument('--finetune-optim',type=json.loads)
parser.add_argument('--finetune-lr-scheduler',type=json.loads)
parser.add_argument('--split-dataset',type=json.loads)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--meta-batch-size', default=128, type=int, metavar='N',
                    help='meta train batchsize')
parser.add_argument('--noise-type', '--nt',default='sym',type=str,choices=['sym','asym'])
parser.add_argument('--noise-rate', '--nr',default=0.0, type=float)

parser.add_argument('--extra',type=str,default='',help='extra information for the exp')

# Miscs
parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--out', default='results',
                        help='Directory to output the result')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)


# TODO expid 需要更加详细的
exp_id = args.arch+args.noise_type+str(args.noise_rate)+args.train_loss+args.trainer+'/'+str(time.time())+'/'
print(exp_id)

# pdb.set_trace()

result_output_path = './result/'+exp_id

if not os.path.isdir(result_output_path):
    mkdir_p(result_output_path)

with open(os.path.join(result_output_path,'augments.json'),'a') as f:
    json.dump(state,f,ensure_ascii=False)
    f.write('\n')

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print(args.gpu)
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


# Data
print(f'==> Preparing relabeled nosiy cifar10')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

datasets = getattr(dataset,args.dataset['type'])(root = './data', args=args,train=True,transform_train=transform_train,transform_val=transform_val, download=True)


 #Construct Model
model_dict = json.loads(args.arch)
model = getattr(model,model_dict['type'])(**model_dict['args'])
model = model.cuda()
# pdb.set_trace()
# model = torch.nn.DataParallel(model).cuda()
cudnn.benchmark = True
print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

# val_criterion = getattr(loss_functions,'ce_loss')

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=args.lr_schedule,gamma=0.1)

title = 'noisy label'
logger = Logger(os.path.join(result_output_path, 'log.txt'), title=title)
logger.set_names(['Learning Rate', 'Train Loss', 'Val Loss', 'Test Loss', 'Train N Acc.', 'Train C Acc', 'Val Acc','Test Acc'])

trainer = getattr(trainers,args.trainer)(model,datasets,logger,result_output_path,args)

# for key,param in trainer.named_parameters():
#     # print(key)
#     print(param)

# pdb.set_trace()
trainer.train()
