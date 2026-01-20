import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class TimeEncoder(nn.Module):
    """
    Time Encoding for TGN to capture continuous time dynamics.
    """
    def __init__(self, dimension, alpha = None, beta= None, parameter_requires_grad: bool = True):
        
        super(TimeEncoder, self).__init__()
        self.dimension = dimension
        self.w = nn.Linear(1, dimension)
        self.w.weight = nn.Parameter((torch.ones(1, dimension)))
        self.w.bias = nn.Parameter((torch.zeros(dimension)))
        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False
            if alpha is None:
                alpha = math.sqrt(dimension)
            if beta is None:
                beta = math.sqrt(dimension)
            self.w.weight = torch.nn.Parameter((torch.from_numpy(1.0 / alpha ** np.linspace(0, dimension/beta, dimension, dtype=np.float32))).reshape(dimension, -1)) 
        else:
            self.w.weight = torch.nn.Parameter((torch.from_numpy(1.0 / 10 ** np.linspace(0, 9, dimension, dtype=np.float32))).reshape(dimension, -1)) 

    def forward(self, t):
        output = torch.cos(self.w(t.float().reshape((-1, 1))))
        return output
