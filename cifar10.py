import numpy as np
from PIL import Image
import pdb
import torch

import torchvision

def get_cifar10_train(root, args, train=True,
                 transform=None,
                 download=False):

    base_dataset = torchvision.datasets.CIFAR10(root, train=train, download=download)
    # pdb.set_trace()
    train_idxs, val_idxs = train_val_split(base_dataset.targets)

    train_dataset = CIFAR10_train(root, train_idxs, val_indices=None, args=args, train=train, transform=transform)
    val_dataset = CIFAR10_val(root, val_idxs, train=train, transform=transform)
    train_val_dataset = CIFAR10_train(root, train_idxs,val_idxs, args, train=train, transform=transform)

    # print (f"Train: {len(train_dataset)} Val: {len(val_dataset)} Train_Val: {len(train_val_dataset)}")
    # pdb.set_trace()
    return train_dataset, val_dataset, train_val_dataset
    

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
        # pdb.set_trace()
        if indexs is not None:
            self.train_data = self.data[indexs]                            #image 本身
            self.train_labels = np.array(self.targets)[indexs]             #noisy label
            self.true_labels = np.array(self.targets)[indexs]              #GT label

        if val_indices is not None:
            self.soft_labels = np.zeros((50000, 10), dtype=np.float32)     #由noisy label产生的soft label
            # self.prediction = np.zeros((50000, 10, 10), dtype=np.float32)
        else:
            self.soft_labels = np.zeros((45000, 10), dtype=np.float32)
            # self.prediction = np.zeros((45000, 10, 10), dtype=np.float32)
        
        self.noisy_idx = []
        self.count = 0
        if args.noise_type == 'asym':
            self.asymmetric_noise()
        elif args.noise_type == 'sym':
            self.symmetric_noise()
        # pdb.set_trace()
        if val_indices is not None:
            self.train_data = np.concatenate((self.train_data,self.data[val_indices]),axis=0)
            self.train_labels = np.concatenate((self.train_labels,np.array(self.targets)[val_indices]),axis=0)
            self.true_labels = np.concatenate((self.true_labels,np.array(self.targets)[val_indices]),axis=0)
        # pdb.set_trace()


        # TODO 这里有问题，因为将trainset中val的label也修改了
        self.generate_soft_labels()    # 由于使用了softmax，这里初始化为1.0
        
    def calculate_real_noisy_rate(self):
        count=0.0
        for j in range(len(self.soft_labels)):
            _, k = torch.max(torch.tensor(self.soft_labels[j]),dim=0)
            if self.true_labels[j] == k:
                count = count + 1
        return count/len(self.soft_labels)    

    def generate_soft_labels(self):
        for i,train_label in enumerate(self.train_labels):
            # self.soft_labels[i][train_label]= 5.0
            self.soft_labels[i] = -1.0
            self.soft_labels[i][train_label] = 1.0

    def symmetric_noise(self):
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.args.noise_rate * len(self.train_data):
                self.train_labels[idx] = np.random.randint(10, dtype=np.int32)
                self.noisy_idx.append(idx)

    def asymmetric_noise(self):
        for i in range(10):
            indices = np.where(self.train_labels == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.args.noise_rate * len(indices):
                    # truck -> automobile
                    if i == 9:
                        self.train_labels[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.train_labels[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.train_labels[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.train_labels[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.train_labels[idx] = 7
                    self.noisy_idx.append(idx)
                # self.soft_labels[idx][self.train_labels[idx]] = 1.   

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
        img, target, soft_target = self.train_data[index], self.train_labels[index], self.soft_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # image;   noisy label; soft label; GT label; index 
        return img, target, soft_target, self.true_labels[index], index




class CIFAR10_val(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_val, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)

        self.data = self.data[indexs]
        self.targets = np.array(self.targets)[indexs]