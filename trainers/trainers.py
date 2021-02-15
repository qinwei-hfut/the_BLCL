from .trainer import Trainer
from .pytrainer import PyTrainer 
from .meta_trainer import MetaTrainer
from .meta_layer_trainer import MetaLayerTrainer
from .neg_trainer import Neg_Trainer
from .DoubleFC_trainer import DoubleFC_Trainer
import pdb

def trainer(*args):
    return Trainer(*args)

def pytrainer(*args):
    return PyTrainer(*args)

def meta_trainer(*args):
    return MetaTrainer(*args)

def meta_layer_trainer(*args):
    return MetaLayerTrainer(*args)

def neg_trainer(*args):
    return Neg_Trainer(*args)

def doubleFC_trainer(*args):
    return DoubleFC_Trainer(*args)

def doubleFC_Test_trainer(*args):
    return DoubleFC_Test_Trainer(*args)