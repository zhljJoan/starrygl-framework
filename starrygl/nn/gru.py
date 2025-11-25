import torch
import torch.nn as nn

from torch import Tensor
from typing import *

class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.r_t = nn.Linear(input_size + hidden_size, hidden_size)
        self.z_t = nn.Linear(input_size + hidden_size, hidden_size)
        self.x_t = nn.Linear(input_size, hidden_size)
        self.h_t = nn.Linear(hidden_size, hidden_size)
    
    @staticmethod
    def gru_input(x: Tensor, h: Tensor) -> Tensor:
        return torch.cat([x, h], dim=-1)
    
    @staticmethod
    def gru_evolve(h: Tensor, r: Tensor, z: Tensor, _x: Tensor, _h: Tensor) -> Tensor:
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(_x + r * _h)
        h = (1 - z) * n + z * h
        return h
    
    def forward(self, x: Tensor, h: Tensor | None = None) -> Tensor:
        if h is None:
            h = torch.zeros(*x.shape[:-1], self.hidden_size, dtype=x.dtype, device=x.device)
        
        _x = self.x_t(x)
        _h = self.h_t(h)
        
        x = self.gru_input(x, h)
        r = self.r_t(x)
        z = self.z_t(x)

        h = self.gru_evolve(h, r, z, _x, _h)
        return h


if __name__ == "__main__":
    cell = GRUCell(32, 96)
    h = None
    for i in range(10):
        x = torch.randn(32)
        h = cell(x, h)
        print(i, h.sum())
