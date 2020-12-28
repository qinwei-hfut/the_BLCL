from .trainer import Trainer
from .pytrainer import PyTrainer 

def trainer(**kwargs):
    return Trainer(**kwargs)

def pytrainer(**kwargs):
    return PyTrainer(**kwargs)