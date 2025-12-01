from asyncio import Queue
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from dgl import DGLGraph
import dgl

from starrygl.data.graph import pyGraph
from starrygl.data.partitions import PartitionData



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

from .partitions import PartitionData



class ChunkState:
    def __init__(self, chunk_index, chunk_count):
        value = torch.stack((torch.arange(chunk_count),chunk_index))
        _value, ind = value.sort(dim = 1)
        self.chunk_nodes = _value[1,:]
        self.chunk_ptr = torch.zeros(chunk_count + 1, dtype=torch.int64)
        self.chunk_ptr[1:] = torch.cumsum(torch.bincount(self.chunk_indicesp[0,:]), dim=0)
        self.order_ind = ind
        
    
class ChunkAwareTemporalLoader:
    """支持chunk划分的数据加载器"""
    
    def __init__(self, data: PartitionData,
                 device:torch.device,
                 stream: torch.cuda.Stream,
                 g: pyGraph,
                 neg_set : Tensor | None = None,):
        self.data = data
        self.device = device
        self.stream = stream if stream is not None else torch.cuda.Stream(device = device)


        self.neg_set = neg_set
        self.g = g
        
        self.load_queue = Queue(maxsize=10)
        
    def is_continuous_time(self):
        return self.data.time_type == 'c'    
    
    def __len__(self):
        if self.is_continuous_time():
            return self.data.num_events
        else:
            return self.data.num_snaps
    
    @staticmethod 
    def generate_chunk_ts_order(type = 'random'):
        time_snaps = []
        chunk_queries = []
        
    def __iter__(self):
        yield from self.__call__()
        
    
    def fetch_subgraph(self, time_queries, chunk_list):
        for time_s, time_e in time_queries:
            subgraph = self.data.load_chunked_subgraph(time_s, time_e, chunk_list)
        return subgraph

    def fetch_khop_subgraph(self, time_queries, chunk_list):
        for time_s, time_e in time_queries:
            subgraph = self.data.load_chunked_khop_subgraph(time_s, time_e, chunk_list
                                                    , device = self.device
                                                    , stream = self.stream
                                                    )
        return subgraph 
    
    def load_chunked_subgraph(self, time_s, time_e, chunk_list):
        
        
    def __call__(self, 
                 chunk_order: Tensor | None = None,
                 time_queries: List[float|int] | None = None,
    ):
        if self.load_queue.empty():
        else:
            yield self.load_queue.popleft()

    
    def chunk_aware_loading(self, chunk_order: Tensor, chunk_batch = 10,
                            time_queries: List[float] = None, 
                            event_index_queries: List[float] = None,
                            
                            ):
        """chunk感知的数据加载"""
        assert time_queries is None or event_index_queries is None
        
        if time_queries is not None:
            for query_time in time_queries:
                for chunk_list in chunk_order.split(chunk_batch):
                    yield self.load_timesnap_subgraph(query_time, chunk_list)
                    
        elif event_index_queries is not None:
            for event_index_query in event_index_queries:
                for chunk_list in chunk_order.split(chunk_batch):
                    yield self.load_eventcount_subgraph(event_index_query, chunk_list)
        else:
            for i in range(self.data.num_snaps):
                for chunk_list in chunk_order.split(chunk_batch):
                    yield self.load_timesnap_subgraph([i,i], chunk_list)
                    

    


