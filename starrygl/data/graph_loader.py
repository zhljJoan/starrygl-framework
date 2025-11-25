import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch import Tensor
from typing import *

import dgl
import dgl.function as fn

from dgl import DGLGraph
from dgl.heterograph import DGLBlock

import math
from torch_sparse import SparseTensor

from starrygl.data.ns_loader import STGraphLoader

from .partitions import PartitionData

__all__ = [
    'GraphLoader'
]
#loader_type 'c'以chunk为中心shuffle else不随机shuffle
class GraphLoader:
    def __init__(self, g: PartitionData,
                 num_snaps: int = 0,
                 device: torch.device = torch.device('cpu'),
                 *args, **kwargs):
        self.g = g
        self.args = args
        self.kwargs = kwargs
    
    @classmethod
    def create_loader(cls, loader_type: str, g: PartitionData, device: str|torch.device,
                        **kwargs) -> 'GraphLoader':
        if loader_type == 'simple':
            return GraphLoader(g, **kwargs)
        elif loader_type == 'STGLoader':
            return STGraphLoader.from_partition_data(g, device, **kwargs)
        elif loader_type == 'NeighborLoader':
            return NeighborLoader.from_partition_data(g, device, **kwargs)
        

    def __len__(self):
        pass
    
    def __iter__(self):
        pass
    
    @classmethod
    def reorder_chunks(self, order:Tensor, score: Tensor|None = None):
        pass
    
    def __getitem__(self, k: slice):
        pass
    
    def __call__(self):
        pass
    
    def __iter__(self):
        pass
    
    def synchronize(self):
        pass
    
    def fetch_graph(self, **args_kwargs):
        pass
                