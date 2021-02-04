from .cifar10 import get_cifar10
from .cifar100 import get_cifar100
from .clothing1m import get_clothing1m

def cifar10(root, args):
    return get_cifar10(root, args)


def cifar100(root, args):
    return get_cifar100(root, args)

def clothing1m(root, args):
    return get_clothing1m(root, args)