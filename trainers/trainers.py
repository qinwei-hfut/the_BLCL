from .trainer import Trainer
from .pytrainer import PyTrainer 
from .meta_trainer import MetaTrainer

def trainer(*args):
    return Trainer(*args)

def pytrainer(*args):
    return PyTrainer(*args)

def meta_trainer(**args):
    return MetaTrainer(**args)