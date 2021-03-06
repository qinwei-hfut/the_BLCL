import os
import torch
import torchvision

import cifar10
import argparse

import pdb
import json
def test_cifar10():
    parser = argparse.ArgumentParser(description='test')
    args = parser.parse_args()
    args.noise_type = 'sym'
    args.noise_rate = 0.8

    train_dataset, val_dataset, train_Cval_dataset, train_Nval_dataset = cifar10.get_cifar10_train(root = './data', args=args,train=True,download=True)

    print(train_dataset.get_noisy_label_acc())
    print(train_dataset.get_soft_labels_acc())

    pdb.set_trace()


test_cifar10()
