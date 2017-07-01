from itertools import chain
import torch
import torchvision as tv
import numpy as np

        
def get_optimizer(model, lr):
    parameters = model.parameters()
    optimizer = torch.optim.Adam(
        parameters, lr=lr, weight_decay=1e-6, betas=(0.9, 0.999))
    return optimizer


