    
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


class NeighborLoader:
    @classmethod
    def from_partition_data(cls,
        data: PartitionData,
        device: str | torch.device,
        chunk_index: Tensor,
        batch_size: int
    ):
        if dist.is_initialized():
            rank = dist.get_rank()
            size = dist.get_world_size()
            group = dist.GroupMember.WORLD
        else:
            rank = 0
            size = 1
            group = None
        
        device = torch.device(device)
        assert device.type == "cuda"

        data = data.pin_memory(device)
        chunk_index = chunk_index.long().to(device)
        chunk_count = chunk_index.max().item() + 1 # 注意这里触发了cuda同步

        # torch.cuda.synchronize(device) # 此时不需要显示调用了

        stream = torch.cuda.Stream(device)

        return cls(
            rank=rank, size=size, group=group,
            data=data, device=device, stream=stream,
            chunk_count=chunk_count,
            chunk_index=chunk_index,
            batch_size = batch_size
        )
    
    @classmethod
    def reorder_chunks(self, order: Tensor, score: Tensor | None = None):
        if score is None:
            return torch.randperm(
                order.numel(),
                dtype=order.dtype,
                device=order.device,
            )
        
        t = score[order]
        t = t.argsort(dim=0, descending=True)
        x = torch.empty_like(t)
        x[t] = torch.arange(t.numel(), dtype=t.dtype, device=t.device)
        return x

    def __init__(self,
        rank: int, size: int, group: Any,
        data: PartitionData,
        device: torch.device,
        stream: torch.cuda.Stream,
        chunk_count: int,
        chunk_index: Tensor,
        batch_size: int
    ) -> None:
        self.rank = rank
        self.size = size
        self.group = group

        self.data = data
        self.device = device
        
        self.chunk_count = chunk_count
        self.chunk_index = chunk_index

        self.num_dst_nodes = chunk_index.numel()
        self.num_snaps = len(self.data)

        self.stream = stream
        self.batch_size = batch_size

        self.iter = 0
    def __len__(self):
        return self.
    
    def __getitem__(self, k: slice):
        assert isinstance(k, slice), "Only slice is supported"
        
        data = self.data[k]
        return type(self)(
            rank=self.rank, size=self.size, group=self.group,
            data=data, device=self.device, stream=self.stream,
            chunk_count=self.chunk_count,
            chunk_index=self.chunk_index,
            
        )
    
    def __iter__(self):
        yield from self.__call__()

    def __call__(self,
        chunk_order: Tensor | None = None,
        chunk_decay: List[int] | None = None,
        w: int = 1,
    ):
        if chunk_order is not None:
            assert chunk_order.numel() == self.chunk_count

            if chunk_decay is None:
                chunk_decay = []

            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.stream):
                chunk_order = chunk_order.to(self.device, non_blocking=True)

                # perm表示本epoch节点顺序, chunk_ends表示分块的end位置
                inds, perm = chunk_order[self.chunk_index].sort(dim=0)
                ends = torch.ops.torch_sparse.ind2ptr(inds, self.chunk_count)

                # nodes_ends表示节点的end位置
                nodes_ends = ends.tolist()
                nodes_ends = [nodes_ends[t] for t in chunk_decay]
        else:
            perm = None
            ends = None
            chunk_decay = []

        seqs = STGraphBlob()
        seqs.max_len = w + len(chunk_decay)

        for i in range(self.num_snaps):
            curr = seqs

            # 加载新的图快照
            with torch.cuda.stream(self.stream):
                g = self.fetch_graph(i, perm=perm, ends=ends) # 获取图快照
            
            # 追加一个图快照
            seqs = seqs._shift(g)

            # 序列不够长，继续加载图快照
            if len(curr) < w:
                continue

            # 衰减时间序列
            if chunk_order is not None:
                with torch.cuda.stream(self.stream):
                    seqs._decay(nodes_ends=nodes_ends, w=w)
            
            if curr is not None:
                self.synchronize()
                curr._ready()
                yield curr
        
        curr = seqs
        seqs = None

        if curr is not None:
            self.synchronize()
            curr._ready()
            yield curr
    
    def synchronize(self):
        if self.stream is not None:
            # 借用barrier完成dist和主stream同步
            work = dist.barrier(async_op=True, group=self.group) if dist.is_initialized() else None

            # 数据stream和主stream同步
            torch.cuda.current_stream(self.device).wait_stream(self.stream)

            # 通信stream和主stream同步
            if work is not None:
                work.wait()
        
    def fetch_graph(self, k: int, perm: Tensor | None = None, ends: Tensor | None = None):
        data = self.data[k].to(device=self.device, non_blocking=True)
        g = data.item(node_perm=perm, keep_ids=True)
        
        if g.route is not None:
            g.route.group = self.group

        if perm is not None:
            assert ends is not None
            g.chunk = ends
        else:
            g.chunk = None

        return g
    

