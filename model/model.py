import torch.nn as nn
import torch.nn.functional as F
from .PreResNet import PreActResNet, PreActBlock
from .ResNet_Zoo import ResNet,BasicBlock

def resnet34(num_classes=10):
    print('construct resnet34')
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)


def PreActResNet18(num_classes=10) -> PreActResNet:
    print("PreActResNet18")
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)

