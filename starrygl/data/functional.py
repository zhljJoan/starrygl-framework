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
from concurrent.futures import ThreadPoolExecutor



def dgl_block_to_graph(g: DGLBlock) -> DGLGraph:
    num_src_nodes = g.num_src_nodes()
    num_dst_nodes = g.num_dst_nodes()
    assert num_src_nodes == num_dst_nodes

    src, dst = g.edges()
    s = dgl.graph(
        (src, dst),
        num_nodes=num_src_nodes,
        idtype=g.idtype,
    )

    for key, val in g.dstdata.items():
        s.ndata[key] = val
    
    for key, val in g.srcdata.items():
        s.ndata[key] = val
    
    for key, val in g.edata.items():
        s.edata[key] = val
    
    return s
