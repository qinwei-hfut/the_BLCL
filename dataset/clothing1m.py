from torchvision import datasets, transforms
import torchvision
import os
import torch


def get_clothing1m(root,args):
    train_transform = transforms.Compose([transforms.Resize((256)),
                                      transforms.RandomCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
                                     ])
    val_transform = transforms.Compose([transforms.Resize((256)),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
                                    ])

    noisy_train_root = os.path.join(root,'noisy_train')
    noisy_train_set = Clothing1M(noisy_train_root,train_transform,True)

    clean_noisy_train_root = os.path.join(root, 'clean_noisy_train')
    clean_noisy_train_set = Clothing1M(clean_noisy_train_root,train_transform, True)
    
    clean_train_root = os.path.join(root, 'clean_train')
    clean_train_set = Clothing1M(clean_train_root,train_transform, True)
    
    clean_val_root = os.path.join(root, 'clean_val')
    clean_val_set = Clothing1M(clean_val_root,val_transform, True)
    
    clean_test_root = os.path.join(root, 'clean_test')
    clean_test_set = Clothing1M(clean_test_root,val_transform, True)


    datasets = {
        'noisy_train_set':noisy_train_set,
        'clean_train_set':clean_train_set,
        'clean_noisy_train_set':clean_noisy_train_set,
        'clean_val_set':clean_val_set,
        'clean_test_set':clean_test_set
    }

    return datasets

class Clothing1M(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, train):
        super(Clothing1M,self).__init__(root,transform)
        self.train = train

    def __getitem__(self,index):
        img, target = super().__getitem__(index)
        if self.train:
            return img, target, torch.tensor(-1), target, index
        else:
            return img, target


# class Clothing1M(VisionDataset):
#     def __init__(self, root, mode='train', transform=None, target_transform=None, num_per_class=-1):

#         super(Clothing1M, self).__init__(root, transform=transform, target_transform=target_transform)

#         if mode=='train':
#             flist = os.path.join(root, "annotations/noisy_train.txt")
#         if mode=='val':
#             flist = os.path.join(root, "annotations/clean_val.txt")
#         if mode=='test':
#             flist = os.path.join(root, "annotations/clean_test.txt")

#         self.impaths, self.targets = self.flist_reader(flist)
        
#         if num_per_class>0:
#             impaths, targets = [], []
#             num_each_class = np.zeros(14)
#             indexs = np.arange(len(self.impaths))                            
#             random.shuffle(indexs)
            
#             for i in indexs:
#                 if num_each_class[self.targets[i]]<num_per_class:
#                     impaths.append(self.impaths[i])
#                     targets.append(self.targets[i])
#                     num_each_class[self.targets[i]]+=1
                    
#             self.impaths, self.targets = impaths, targets
#             print('#samples/class: {};\n#total samples: {:d}\n'.format([int(i) for i in num_each_class], int(sum(num_each_class))))

# #         # for quickly ebug
# #         self.impaths, self.targets = self.impaths[:1000], self.targets[:1000]


#     def __getitem__(self, index):
#         impath = self.impaths[index]
#         target = self.targets[index]

#         img = Image.open(impath).convert("RGB")

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target

#     def __len__(self):
#         return len(self.impaths)

#     def flist_reader(self, flist):
#         impaths = []
#         targets = []
#         with open(flist, 'r') as rf:
#             for line in rf.readlines():
#                 row = line.split(" ")
#                 impaths.append(self.root + '/' + row[0])
#                 targets.append(int(row[1]))
#         return impaths, targets
