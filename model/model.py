import torch.nn as nn
import torch.nn.functional as F
from .PreResNet import PreActResNet, PreActBlock
from .ResNet_Zoo import resnet50,resnet34
from .PyPreResNet import PyPreActResNet, PyPreActBlock
from .ToyModel import ToyModel
from .Multi_FC_Model import Multi_FC_Model

def ResNet50(num_classes=14):
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def PyPreActResNet18(num_classes=10) -> PreActResNet:
    print('PyPreActResNet18')
    return PyPreActResNet(PyPreActBlock, [2,2,2,2], num_classes=num_classes)

def PreActResNet18(num_classes=10) -> PreActResNet:
    print("PreActResNet18")
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)

def toymodel():
    return ToyModel()

def Multi_FC_PreActResNet18(num_classes=10) -> PreActResNet:
    print('Multi_FC_PreActResNet18')
    model = PreActResNet18(num_classes=num_classes)
    return Multi_FC_Model(model=model,num_classes=num_classes)


