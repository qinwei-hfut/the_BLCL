import sys

import numpy as np
from PIL import Image
import torchvision
from torch.utils.data.dataset import Subset
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances 
import torch
import torch.nn.functional as F
import random 
import os
import json
from numpy.testing import assert_array_almost_equal



def get_cifar100(root, args, train=True,
                transform_train=None, transform_val=None,
                download=False):
    base_dataset = torchvision.datasets.CIFAR100(root, train=train, download=download)
    train_idxs, val_idxs = train_val_split(base_dataset.targets)

    train_Nval_dataset = CIFAR100_train(root, train_idxs+val_idxs, val_indices=None, args=args, train=train, transform=transform_train)
    train_dataset = CIFAR100_train(root, train_idxs, val_indices=None, args=args, train=train, transform=transform_train)
    val_dataset = CIFAR100_train(root, val_idxs, val_indices=None, args=args, train=train, transform=transform_val)
    train_Cval_dataset = CIFAR100_train(root, train_idxs,val_idxs, args, train=train, transform=transform_train)

    testset = torchvision.datasets.CIFAR100(root, train=False, transform=transform_val)
    
    
    return train_dataset, val_dataset, train_Cval_dataset,train_Nval_dataset,testset


def train_val_split(train_val):
    num_classes = 100
    train_val = np.array(train_val)
    train_n = int(len(train_val) * 0.9 / num_classes)
    train_idxs = []
    val_idxs = []

    for i in range(num_classes):
        idxs = np.where(train_val == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:train_n])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs


class CIFAR100_train(torchvision.datasets.CIFAR100):
    def __init__(self, root, indexs=None, val_indices=None, args=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100_train, self).__init__(root, train=train,
                                            transform=transform, target_transform=target_transform,
                                            download=download)
        self.num_classes = 100
        self.args = args
        if indexs is not None:
            self.train_data = self.data[indexs]                            #image 本身
            self.noisy_labels = np.array(self.targets)[indexs]             #noisy label
            self.gt_labels = np.array(self.targets)[indexs]              #GT label

        if val_indices is not None:
            self.soft_labels = np.zeros((len(indexs)+len(val_indices),self.num_classes),dtype=np.float32)
        else:
            self.soft_labels = np.zeros((len(indexs),self.num_classes),dtype=np.float32)

        self.noisy_idx = []
        self.add_noise()

        if val_indices is not None:
            self.train_data = np.concatenate((self.train_data,self.data[val_indices]),axis=0)
            self.noisy_labels = np.concatenate((self.noisy_labels,np.array(self.targets)[val_indices]),axis=0)
            self.gt_labels = np.concatenate((self.gt_labels,np.array(self.targets)[val_indices]),axis=0)

        self.init_soft_labels_from_constant_values()

    def add_noise(self):
        if self.args.noise_type == 'asym':
            self.asymmetric_noise()
        elif self.args.noise_type == 'sym':
            self.symmetric_noise()

    def get_soft_labels_acc(self):
        return (np.argmax(self.soft_labels,axis=1)  == self.gt_labels).sum() / len(self.gt_labels)

    def get_noisy_label_acc(self):
        return sum(self.noisy_labels == self.gt_labels)/len(self.gt_labels)

    def init_soft_labels_from_noisy_labels(self):
        self.soft_labels = torch.tensor(self.soft_labels).scatter(1,torch.tensor(self.targets).view(-1,1),1).numpy()

    def init_soft_labels_from_constant_values(self):
        for i,train_label in enumerate(self.noisy_labels):
            self.soft_labels[i] = -1.0
            self.soft_labels[i][train_label] = 1.0

    def symmetric_noise(self):
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.args.noise_rate * len(self.train_data):
                self.noisy_idx.append(idx)
                self.noisy_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)

    def multiclass_noisify(self, y, P, random_state=0):
        """ Flip classes according to transition probability matrix T.
        It expects a number between 0 and the number of classes - 1.
        """

        assert P.shape[0] == P.shape[1]
        assert np.max(y) < P.shape[0]

        # row stochastic matrix
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
        assert (P >= 0.0).all()

        m = y.shape[0]
        new_y = y.copy()
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = y[idx]
            # draw a vector with only an 1
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]

        return new_y

#     def build_for_cifar100(self, size, noise):
#         """ random flip between two random classes.
#         """
#         assert(noise >= 0.) and (noise <= 1.)

#         P = np.eye(size)
#         cls1, cls2 = np.random.choice(range(size), size=2, replace=False)
#         P[cls1, cls2] = noise
#         P[cls2, cls1] = noise
#         P[cls1, cls1] = 1.0 - noise
#         P[cls2, cls2] = 1.0 - noise

#         assert_array_almost_equal(P.sum(axis=1), 1, 1)
#         return P
    def build_for_cifar100(self, size, noise):
        """ The noise matrix flips to the "next" class with probability 'noise'.
        """

        assert(noise >= 0.) and (noise <= 1.)

        P = (1. - noise) * np.eye(size)
        for i in np.arange(size - 1):
            P[i, i + 1] = noise

        # adjust last row
        P[size - 1, 0] = noise

        assert_array_almost_equal(P.sum(axis=1), 1, 1)
        return P


    def asymmetric_noise(self):
        P = np.eye(self.num_classes)
        n = self.args.noise_rate
        nb_superclasses = 20
        nb_subclasses = 5

        if n > 0.0:
            for i in np.arange(nb_superclasses):
                init, end = i * nb_subclasses, (i+1) * nb_subclasses
                P[init:end, init:end] = self.build_for_cifar100(nb_subclasses, n)

            y_train_noisy = self.multiclass_noisify(self.noisy_labels, P=P,
                                               random_state=0)
            actual_noise = (y_train_noisy != self.noisy_labels).mean()
            assert actual_noise > 0.0
            self.noisy_labels = y_train_noisy
            
            
    def update_data(self, indices, updated_soft_targets):
        indices = indices.cpu()
        updated_soft_targets = updated_soft_targets.detach().cpu()
        for i,idx in enumerate(indices):
            self.soft_labels[idx] = updated_soft_targets[i]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, noisy_targets = self.train_data[index], self.noisy_labels[index] 
        soft_targets, gt_targets = self.soft_labels[index], self.gt_labels[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)


        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, noisy_targets, soft_targets, gt_targets, index

    def __len__(self):
        return len(self.train_data)


    
