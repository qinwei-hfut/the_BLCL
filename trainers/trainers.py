from .trainer import Trainer
from .pytrainer import PyTrainer 

def trainer(*args):
    return Trainer(*args)

def pytrainer(*args):
    return PyTrainer(*args)