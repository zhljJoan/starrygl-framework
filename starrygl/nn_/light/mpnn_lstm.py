import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *

from starrygl.data import 

from ..lstm import LSTMCell
from ..graphconv import GCN



class MPNN_LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.gcn = GCN(input_size, hidden_size, num_layers=2)
        self.rnn = nn.ModuleList([
            LSTMCell(hidden_size, hidden_size),
            LSTMCell(hidden_size, hidden_size),
        ])
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, seqs: AsyncGraphBatch) -> List[Tensor]:
        outs: List[Tensor] = []

        inps: List[Tensor] = self.gcn.layerwise(seqs.values())

        h = None
        for k, x in zip(seqs, inps):
            h = seqs.pull_graph_state(k, h)

            s1 = None if h is None else h[0:2]
            s2 = None if h is None else h[2:4]

            x, _ = s1 = cast(LSTMCell, self.rnn[0]).forward(x, s1)
            x, _ = s2 = cast(LSTMCell, self.rnn[1]).forward(x, s2)

            h = s1 + s2
            
            seqs.push_graph_state(k, h)
            outs.append(x)

        outs = [self.out(F.relu(x)) for x in outs]
        return outs
    
