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

import dataset.cifar10 as cifar10
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

# from trainers.trainer import Trainer
import trainers.trainer as trainer
import model.PreResNet as models


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
parser.add_argument('--label', default='result',
                        help='Directory to input the labels')
# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise-type', '--nt',default='sym',type=str,choices=['sym','asym'])
parser.add_argument('--noise-rate', '--nr',default=0.0, type=float)

# Miscs
parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--out', default='results',
                        help='Directory to output the result')

args = parser.parse_args()
args.lr_schedule = [40,80]

state = {k: v for k, v in args._get_kwargs()}
print(state)

# TODO expid 需要更加详细的
exp_id = args.noise_type+str(args.noise_rate)+'/'+str(time.time())+'/'
print(exp_id)

result_output_path = './result/'+exp_id

if not os.path.isdir(result_output_path):
    mkdir_p(result_output_path)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
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

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset, val_dataset, train_Cval_dataset, train_Nval_dataset = cifar10.get_cifar10_train(root = './data', args=args,train=True,transform_train=transform_train,transform_val=transform_val, download=True)
testset = torchvision.datasets.CIFAR10('./data', train=False, transform=transform_test)
datasets = train_dataset, val_dataset, train_Cval_dataset, train_Nval_dataset, testset

 # Model
print("==> creating preact_resnet")
model = models.PreActResNet18()
model = model.cuda()
# model = torch.nn.DataParallel(model).cuda()
cudnn.benchmark = True
print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

ce_criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

title = 'noisy label'
logger = Logger(os.path.join(result_output_path, 'log.txt'), title=title)
logger.set_names(['Learning Rate', 'Train Loss', 'Test Loss', 'Train N Acc.', 'Train C Acc', 'Test Acc'])

trainer = trainer.Trainer(model,datasets,optimizer,ce_criterion,logger,result_output_path,args)
trainer.train()

def soft_ce(outputs, soft_targets):
    loss = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * soft_targets, dim=1))
    return loss