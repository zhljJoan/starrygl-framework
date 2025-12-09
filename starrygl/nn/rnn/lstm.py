import torch
import torch.nn as nn

from torch import Tensor
from typing import *



class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i_t = nn.Linear(input_size + hidden_size, hidden_size)
        self.f_t = nn.Linear(input_size + hidden_size, hidden_size)
        self.g_t = nn.Linear(input_size + hidden_size, hidden_size)
        self.o_t = nn.Linear(input_size + hidden_size, hidden_size)

    @staticmethod
    def lstm_input(x: Tensor, h: Tensor) -> Tensor:
        return torch.cat([x, h], dim=-1)
    
    @staticmethod
    def lstm_evolve(c: Tensor, i: Tensor, f: Tensor, g: Tensor, o: Tensor) -> Tuple[Tensor, Tensor]:
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c
    
    def forward(self, x: Tensor, state: Tuple[Tensor, Tensor] | None = None) -> Tuple[Tensor, Tensor]:
        if state is None:
            h = torch.zeros(*x.shape[:-1], self.hidden_size, dtype=x.dtype, device=x.device)
            c = torch.zeros_like(h)
        else:
            h, c = state

        x = self.lstm_input(x, h)
        
        i = self.i_t(x)
        f = self.f_t(x)
        g = self.g_t(x)
        o = self.o_t(x)

        h, c = self.lstm_evolve(c, i, f, g, o)
        return h, c

if __name__ == "__main__":
    cell = LSTMCell(32, 96)
    state = None
    for i in range(10):
        x = torch.randn(32)
        h, c = state = cell(x, state)
        print(i, h.sum(), c.sum())
        