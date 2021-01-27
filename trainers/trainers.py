from .trainer import Trainer
from .pytrainer import PyTrainer 
from .meta_trainer import MetaTrainer
from .meta_layer_trainer import MetaLayerTrainer
import pdb

def trainer(*args):
    return Trainer(*args)

def pytrainer(*args):
    return PyTrainer(*args)

def meta_trainer(*args):
    return MetaTrainer(*args)

def meta_layer_trainer(*args):
    return MetaLayerTrainer(*args)