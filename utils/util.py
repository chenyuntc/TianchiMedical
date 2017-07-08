from itertools import chain
import torch
import torchvision as tv
import numpy as np
from .yellowfin import YFOptimizer

        
def get_optimizer(model, lr,weight_decay=1e-5):
    parameters = model.parameters()
    optimizer = torch.optim.Adam(
        parameters, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    return optimizer


def get_yellow(model,lr):
    return YFOptimizer(model.parameters(),weight_decay=1e-4)