
class Clothing1M(VisionDataset):
    def __init__(self, root, mode='train', transform=None, target_transform=None, num_per_class=-1):

        super(Clothing1M, self).__init__(root, transform=transform, target_transform=target_transform)

        if mode=='train':
            flist = os.path.join(root, "annotations/noisy_train.txt")
        if mode=='val':
            flist = os.path.join(root, "annotations/clean_val.txt")
        if mode=='test':
            flist = os.path.join(root, "annotations/clean_test.txt")

        self.impaths, self.targets = self.flist_reader(flist)
        
        if num_per_class>0:
            impaths, targets = [], []
            num_each_class = np.zeros(14)
            indexs = np.arange(len(self.impaths))                            
            random.shuffle(indexs)
            
            for i in indexs:
                if num_each_class[self.targets[i]]<num_per_class:
                    impaths.append(self.impaths[i])
                    targets.append(self.targets[i])
                    num_each_class[self.targets[i]]+=1
                    
            self.impaths, self.targets = impaths, targets
            print('#samples/class: {};\n#total samples: {:d}\n'.format([int(i) for i in num_each_class], int(sum(num_each_class))))

#         # for quickly ebug
#         self.impaths, self.targets = self.impaths[:1000], self.targets[:1000]


    def __getitem__(self, index):
        impath = self.impaths[index]
        target = self.targets[index]

        img = Image.open(impath).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.impaths)

    def flist_reader(self, flist):
        impaths = []
        targets = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                row = line.split(" ")
                impaths.append(self.root + '/' + row[0])
                targets.append(int(row[1]))
        return impaths, targets
