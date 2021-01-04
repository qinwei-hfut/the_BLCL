from .cifar10 import get_cifar10
from .cifar100 import get_cifar100

def cifar10(root, args, train=True,
                 transform_train=None,
                 transform_val=None,
                 download=False):
    return get_cifar10(root, args, train,transform_train,transform_val,download)


def cifar100(root, args, train=True,
                 transform_train=None,
                 transform_val=None,
                 download=False):
    return get_cifar100(root, args, train,transform_train,transform_val,download)