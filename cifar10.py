import numpy as np
from PIL import Image
import pdb
import torch

import torchvision

def get_cifar10_train(root, args, train=True,
                 transform_train=None,
                 transform_val=None,
                 download=False):

    base_dataset = torchvision.datasets.CIFAR10(root, train=train, download=download)
    # pdb.set_trace()
    train_idxs, val_idxs = train_val_split(base_dataset.targets)

    train_Nval_dataset = CIFAR10_train(root, train_idxs+val_idxs, val_indices=None, args=args, train=train, transform=transform_train)
    train_dataset = CIFAR10_train(root, train_idxs, val_indices=None, args=args, train=train, transform=transform_train)
    val_dataset = CIFAR10_train(root, val_idxs, val_indices=None, args=args, train=train, transform=transform_val)
    train_Cval_dataset = CIFAR10_train(root, train_idxs,val_idxs, args, train=train, transform=transform_train)

    # train_dataset:仅仅45K的noisy trainset(但是noisy label; gt label; soft label都输出)
    # val_dataset: 5K的noisy val set，同样，由于同时输出noisy label，gt label，soft label，可以既满足clean val进行val，可以满足使用noisy val进行val
    # train_Nval_dataset：50K张图片都输出，同样也输出noisy label, gt label, soft label;但是需要注意的是被包含的val的noisy label其实是noisy的
    # train_Cval_dataset: 50K张图片都输出，同样也输出noisy label，gt label，soft label；但是需要注意的是被包含的val的noisy label其实都是clean的
    # 注意：这些dataset的train和val的分割是一样的；但是，noisy label examples的选择肯定是不一样的；
    return train_dataset, val_dataset, train_Cval_dataset,train_Nval_dataset
    

def train_val_split(train_val):
    train_val = np.array(train_val)
    train_n = int(len(train_val) * 0.9 / 10)
    train_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(train_val == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:train_n])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs

class CIFAR10_train(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, val_indices=None, args=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_train, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.args = args
        if indexs is not None:
            self.train_data = self.data[indexs]                            #image 本身
            self.noisy_labels = np.array(self.targets)[indexs]             #noisy label
            self.gt_labels = np.array(self.targets)[indexs]              #GT label

        if val_indices is not None:
            # self.soft_labels = np.zeros((50000, 10), dtype=np.float32)     #由noisy label产生的soft label
            self.soft_labels = np.zeros((len(indexs)+len(val_indices),10),dtype=np.float32)
        else:
            # self.soft_labels = np.zeros((45000, 10), dtype=np.float32)
            self.soft_labels = np.zeros((len(indexs),10),dtype=np.float32)
        
        self.noisy_idx = []
        self.add_noise()
        
        if val_indices is not None:
            self.train_data = np.concatenate((self.train_data,self.data[val_indices]),axis=0)
            self.noisy_labels = np.concatenate((self.noisy_labels,np.array(self.targets)[val_indices]),axis=0)
            self.gt_labels = np.concatenate((self.gt_labels,np.array(self.targets)[val_indices]),axis=0)

        self.init_soft_labels_from_constant_values()
        # self.init_soft_labels_from_noisy_labels()
        
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
                self.noisy_labels[idx] = np.random.randint(10, dtype=np.int32)
                # 由于random类别，可能到自己，idx不一定都是标错到
                self.noisy_idx.append(idx)

    def asymmetric_noise(self):
        for i in range(10):
            indices = np.where(self.noisy_labels == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.args.noise_rate * len(indices):
                    # truck -> automobile
                    if i == 9:
                        self.noisy_labels[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.noisy_labels[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.noisy_labels[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.noisy_labels[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.noisy_labels[idx] = 7
                    self.noisy_idx.append(idx)
                # self.soft_labels[idx][self.noisy_labels[idx]] = 1.   

    def update_data(self, indices, updated_soft_targets):
        indices = indices.cpu()
        updated_soft_targets = updated_soft_targets.detach().cpu()
        for i,idx in enumerate(indices):
            self.soft_labels[idx] = updated_soft_targets[i]

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, noisy_target = self.train_data[index], self.noisy_labels[index] 
        soft_targets, gt_targets = self.soft_labels[index], self.gt_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        # image;   noisy label; soft label; GT label; index 
        return img, noisy_target, soft_targets, gt_targets, index




