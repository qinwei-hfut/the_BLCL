from torchvision import datasets, transforms
import torchvision
import os
import torch
import random

from torchvision.datasets.vision import VisionDataset

from PIL import Image

import os
import os.path
import sys

import torch
import torchvision.transforms as transforms
import random
from datetime import datetime

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
    
    clean_train_root = os.path.join(root, '100_clean_train')
    clean_train_set = Clothing1M(clean_train_root,train_transform, True)
    
    clean_val_root = os.path.join(root, '100_clean_val')
    clean_val_set = Clothing1M(clean_val_root,val_transform, True)
    
    clean_test_root = os.path.join(root, 'clean_test')
    clean_test_set = Clothing1M(clean_test_root,val_transform, False)


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

'''
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)



def make_dataset(dir_image, class_to_idx, sample_per_class, extensions=None, is_valid_file=None):
    # print('make')

    images = []
    dir_image = os.path.expanduser(dir_image)
    # dir_sal = os.path.expanduser(dir_sal)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    idx = 0

    # print(class_to_idx)
    for target in sorted(class_to_idx.keys()):

        
        idx = idx + 1

        d_image = os.path.join(dir_image, target)
        # d_sal = os.path.join(dir_sal, target)
        # print('-------------------------------------------------------------------------------')
        # print(target)
        if not os.path.isdir(d_image):
            continue
        # if not os.path.isdir(d_sal):
        #     continue
        sample_per_class_count = 0
        for root, _, fnames in sorted(os.walk(d_image, followlinks=True)):
            # TODO random frames //////no shuffle; should sorted
            random.shuffle(fnames)
            for fname in fnames:

                # print(fname)
                image_path = os.path.join(d_image, fname)
                # sal_path = os.path.join(d_sal, fname)
                if is_valid_file(image_path):
                    item = (image_path, class_to_idx[target])
                    images.append(item)
                    sample_per_class_count = sample_per_class_count + 1
                    if sample_per_class_count == sample_per_class:
                        break

    return images


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root_image, loader, sample_per_class, extensions=None, transform_img=None, 
                        target_transform=None, is_valid_file=None):
        super(DatasetFolder, self).__init__(transform=transform_img,
                                            target_transform=target_transform)
        self.root_image = root_image

        classes, class_to_idx = self._find_classes(self.root_image)
        samples = make_dataset(self.root_image, class_to_idx, sample_per_class, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))
        self.transform_img = transform_img
        # self.transform_sal = transform_sal
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[2] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        image_path, target = self.samples[index]
        image = self.loader(image_path)
        # sal = self.pil_L_loader(sal_path)

        # sal_tensor = transforms.ToTensor()(transforms.Resize([224,224])(sal))
        

    
        # seed = datetime.now()
        # random.seed(seed)
        if self.transform_img is not None:
            image_tensor = self.transform_img(image)
        # random.seed(seed)
        # if self.transform_sal is not None:
        #     sal_tensor = self.transform_sal(sal)
        if self.target_transform is not None:
            target = self.target_transform(target)

        

        return image_tensor, target

    def __len__(self):
        return len(self.samples)

    # def pil_L_loader(self,path):
    #     with open(path, 'rb') as f:
    #         img = Image.open(f)
    #         return img.convert('L')


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp','.xml')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



def pil_1_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('1')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root_image,sample_per_class, transform_img=None, 
                 target_transform=None, loader=default_loader, is_valid_file=None):
        super(ImageFolder, self).__init__(root_image, loader, sample_per_class, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform_img=transform_img,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples



class Clothing1M(ImageFolder):
    def __init__(self, root,transform, train, sample_per_class=float('inf')):
        super(Clothing1M,self).__init__(root,sample_per_class,transform)
        self.train = train

    def __getitem__(self,index):
        img, target = super().__getitem__(index)
        if self.train:
            return img, target, torch.tensor(-1), target, index
        else:
            return img, target
'''