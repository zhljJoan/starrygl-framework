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


__all__ = [
    'STGraphBlob',
    'STGraphLoader',
]


class STNodeState:
    def __init__(self, state: Tensor | Tuple[Tensor,...] | None = None):
        if state is None:
            self.data = None
            self.type = None
        elif isinstance(state, Tensor):
            self.data = state.detach()
            self.type = "tensor"
        else:
            self.data = tuple(s.detach() for s in state)
            self.type = "tuple"
    
    def compose(self, g: DGLBlock, state: Tensor | Tuple[Tensor,...] | None):
        n = g.num_dst_nodes()
        if self.data is None:
            if state is None:
                return None
            elif isinstance(state, Tensor):
                return self._pad(state, n)
            else:
                return tuple(self._pad(s, n) for s in state)
        else:
            if state is None:
                if isinstance(self.data, Tensor):
                    return self._pad(self.data, n)
                else:
                    return tuple(self._pad(s, n) for s in self.data)
            elif isinstance(state, Tensor):
                assert isinstance(self.data, Tensor)
                return self._mix(state, self.data, n)
            else:
                assert not isinstance(self.data, Tensor)
                assert len(state) == len(self.data)
                return tuple(self._mix(s, _s, n) for s, _s in zip(state, self.data))
    
    @staticmethod
    def _pad(x: Tensor, n: int) -> Tensor:
        m = x.size(0)
        if m < n: # pad with zeros
            padding = x.new_zeros(n - m, *x.shape[1:])
            return torch.cat([x, padding], dim=0)
        elif m == n: # no padding needed
            return x
        else: # truncate to size n
            return x[:n]
    
    @staticmethod
    def _mix(x: Tensor, _x: Tensor, n: int) -> Tensor:
        _m = _x.size(0) # old state size
        assert _m >= n, f"Old state size {_m} must be at least {n} for mixing"

        m = x.size(0)   # new state to mix with old state
        if m > n: # truncate to size n
            return x[:n]
        elif m == n:
            return x
        else:
            return torch.cat([x, _x[m:n]], dim=0)

class STGraphBlob:
    def __init__(self):
        self.graphs: Dict[int, DGLBlock | DGLGraph] = {}
        self.buffer: Dict[int, STNodeState] = {}

        self.max_len: int = 0
        self.cur_end: int = 0
        self.last_buf: Dict[int, STNodeState] = {}
        self.next_buf: Dict[int, STNodeState] = {}
    
    def __len__(self):
        return len(self.graphs)
    
    def __iter__(self):
        yield from self.keys()
    
    def __repr__(self):
        ss = []
        for k in self:
            g = self.graphs[k]
            if g.is_block:
                ss.append(f"[{k}]{{{g.num_src_nodes()}->{g.num_dst_nodes()}}}")
            else:
                ss.append(f"[{k}]{{{g.num_nodes()}}}")
        ss = " ".join(ss)
        return f"{type(self).__name__}({ss})"
    
    def keys(self):
        yield from sorted(self.graphs.keys())
    
    def values(self):
        for k in self.keys():
            yield self.get_block(k)

    def items(self):
        for k in self.keys():
            g = self.get_block(k)
            yield k, g
    
    def end_key(self):
        return self.cur_end - 1
    
    def get_loss_scale(self, enable: bool = True) -> float:
        if not enable:
            return 1.0
        s = self.get_grads_scale("linear")
        return 1.0 / s
    
    def get_grads_scale(self, mode: str = "none") -> Tuple[float, float]:
        mode = mode.lower()
        if mode == "none":
            return 1.0, 1.0
        
        mode, tag = mode.split(":")
        
        n = self.graphs[self.end_key()].num_dst_nodes()

        m = 0
        s = 0
        for k, g in self.graphs.items():
            if k == self.end_key():
                continue
            m += g.num_dst_nodes()
            s += 1
        
        if m <= 0:
            return 1.0, 1.0
        s = n * s / m

        if mode == "sqrt":
            s = s ** 0.5
        elif mode == "log":
            assert s >= 1.0
            s = math.log(s)
        elif mode == "linear":
            s = s
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if tag == "grad":
            return s, 1.0
        elif tag == "loss":
            return 1.0, 1.0 / s
        else:
            raise ValueError(f"Unknown tag: {tag}")
    
    def erase_states_(self, enable: bool = True):
        if not enable:
            return self
        self.buffer.clear()
        for k in self.graphs.keys():
            self.buffer[k] = STNodeState()
        return self
    
    def erase_routes_(self, enable: bool = True):
        if not enable:
            return self
        for k in list(self.graphs.keys()):
            g = self.graphs[k]
            if not g.is_block:
                continue
            t = self.truncate_block_srcnodes(g)
            t.route = None
            t.chunk = g.chunk
            self.graphs[k] = t
        return self

    def get_block(self, k: int | None = None):
        if k is None:
            assert len(self.graphs) == 1
            for g in self.graphs.values():
                return g
        return self.graphs[k]

    def pull_graph_state(self, k: int, state: Tensor | Tuple[Tensor,...] | None = None, erase: bool = True):
        g = self.graphs[k] # DGLBlock
        s = self.buffer[k] # historial state

        state = s.compose(g, state)
        if erase:
            self.buffer.pop(k)
        return state
    
    def push_graph_state(self, k: int, state: Tensor | Tuple[Tensor,...] | None = None):
        assert k in self.graphs, f"graph {k} not in blob"
        self.next_buf[k + 1] = STNodeState(state)
        return state

    def _shift(self, g: DGLBlock):
        this = type(self)()
        this.max_len = self.max_len
        this.cur_end = self.cur_end + 1
        this.last_buf = self.next_buf
        
        this.graphs[this.cur_end - 1] = g

        k = this.cur_end - 2
        while k in self.graphs and len(this.graphs) < this.max_len:
            this.graphs[k] = self.graphs[k]
            k -= 1
        return this
    
    def _ready(self):
        for k in self.graphs:
            self.buffer[k] = self.last_buf.get(k, STNodeState())
        self.last_buf = {} # 这里要删除引用，释放显存

    def _decay(self, nodes_ends: List[int], w: int = 1):
        for k in self:
            g = self.graphs[k]

            # 计算与leader快照的偏移量
            p = self.cur_end - k - 1 - w
            if p < 0:
                continue
            
            g = self.truncate_block_to_graph(g, end=nodes_ends[p])
            # g.route = None # 设置route为None表示不再进行分布式同步
            # g.chunk = None # 设置chunk为None表示不再进行chunk划分

            self.graphs[k] = g
    
    @classmethod
    def truncate_block_srcnodes(cls, g: DGLBlock):
        assert g.is_block
        num_src_nodes = g.num_src_nodes()
        num_dst_nodes = g.num_dst_nodes()
        assert num_src_nodes >= num_dst_nodes

        src, dst = g.edges()
        m = src < num_dst_nodes
        src = src[m]
        dst = dst[m]

        s = dgl.create_block(
            (src, dst),
            num_src_nodes=num_dst_nodes,
            num_dst_nodes=num_dst_nodes,
            idtype=g.idtype,
        )

        for key, val in g.dstdata.items():
            s.dstdata[key] = val

        for key, val in g.srcdata.items():
            s.srcdata[key] = val[:num_dst_nodes].clone()

        for key, val in g.edata.items():
            if key == 'gcn_norm':
                continue
            s.edata[key] = val[m]

    @classmethod
    def truncate_block_to_graph(cls, g: DGLBlock | DGLGraph, end: int):
        if g.is_block:
            num_src_nodes = g.num_src_nodes()
            num_dst_nodes = g.num_dst_nodes()
            assert num_src_nodes >= num_dst_nodes
            
            assert end <= num_dst_nodes, "end must be less than or equal to num_dst_node"

            src, dst = g.edges()
            m = (src < end) & (dst < end)
            src = src[m]
            dst = dst[m]

            s = dgl.graph(
                (src, dst),
                num_nodes=end,
                idtype=g.idtype,
            )

            for key, val in g.dstdata.items():
                s.ndata[key] = val[:end].clone()

            for key, val in g.srcdata.items(): # src特征优先
                s.ndata[key] = val[:end].clone()

            for key, val in g.edata.items():
                if key == 'gcn_norm':
                    continue
                s.edata[key] = val[m]
        
        else:
            num_nodes = g.num_nodes()
            assert end <= num_nodes, "end must be less than or equal to num_nodes"

            src, dst = g.edges()
            m = (src < end) & (dst < end)
            src = src[m]
            dst = dst[m]
            
            s = dgl.graph(
                (src, dst),
                num_nodes=end,
                idtype=g.idtype,
            )

            for key, val in g.ndata.items():
                s.ndata[key] = val[:end].clone()
            
            for key, val in g.edata.items():
                if key == 'gcn_norm':
                    continue
                s.edata[key] = val[m]

        return s
    
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
                    

    


