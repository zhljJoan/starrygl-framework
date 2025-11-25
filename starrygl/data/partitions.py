import torch
import torch.nn as nn
import torch.distributed as dist

from torch import Tensor
from typing import *

from pathlib import Path

import dgl
import dgl.function as fn
from dgl.heterograph import DGLBlock

from .collection import TensorData, RouteData


__all__ = [
    'PartitionData',
]


class PartitionData:
    def __init__(self,
        src_ids: TensorData,
        dst_ids: TensorData,
        edge_ids: TensorData,
        edge_src: TensorData,
        edge_dst: TensorData,
        routes: RouteData | None = None,
        node_data: Dict[str, TensorData] = {},
        edge_data: Dict[str, TensorData] = {},
    ):
        self.src_ids = src_ids
        self.dst_ids = dst_ids
        self.edge_ids = edge_ids
        self.edge_src = edge_src
        self.edge_dst = edge_dst

        self.routes = routes
        self.node_data = node_data
        self.edge_data = edge_data

        self.num_events = len(self.dst_ids.data)
        self.num_snaps = len(self.dst_ids)
        self.num_dst_nodes = self.dst_ids[0].item().numel()
        
        n = self.num_dst_nodes
        for i in range(self.num_snaps):
            m = self.dst_ids[i].item().numel()
            assert m == n, f"Expected {n} dst_ids[{i}], got {m}"
        
        n = self.num_snaps
        for key, val in self.node_data.items():
            assert len(val) == n, f"Expected {n} node_data[{key}], got {len(val)}"
        
        for key, val in self.edge_data.items():
            assert len(val) == n, f"Expected {n} edge_data[{key}], got {len(val)}"

        for key, val in self.__dict__.items():
            if not isinstance(val, TensorData):
                continue
            assert len(val) == n, f"Expected {n} {key}, got {len(val)}"

    def __getstate__(self):
        state = {
            "routes": self.routes,
            "node_data": self.node_data,
            "edge_data": self.edge_data,
        }

        for key, val in self.__dict__.items():
            if not isinstance(val, TensorData):
                continue
            state[key] = val
        return state
    
    def __setstate__(self, state):
        for key, val in state.items():
            setattr(self, key, val)
    
    def __repr__(self) -> str:
        n = len(self)
        return f"{type(self).__name__}(len={n})"
    
    def __len__(self):
        return self.num_snaps
    
    def __getitem__(self, k: int | slice):
        state = {
            "node_data": {_k: v[k] for _k, v in self.node_data.items()},
            "edge_data": {_k: v[k] for _k, v in self.edge_data.items()},
        }

        if self.routes is None:
            state["routes"] = None
        else:
            state["routes"] = self.routes[k]

        for key, val in self.__dict__.items():
            if not isinstance(val, TensorData):
                continue
            state[key] = val[k]
        return type(self)(**state)
    
    def node_keys(self):
        return self.node_data.keys()
    
    def add_ndata(self, key: str, data: TensorData):
        assert isinstance(data, TensorData), f"{data} is not a TensorData"
        assert len(data) == len(self), f"self requires {len(self)} elements, but {key} has {len(data)} elements"
        self.node_data[key] = data
    
    def pop_ndata(self, key: str) -> Optional[TensorData]:
        return self.node_data.pop(key, None)
    
    def edge_keys(self):
        return self.edge_data.keys()
    
    def add_edata(self, key: str, data: TensorData):
        assert isinstance(data, TensorData), f"{data} is not a TensorData"
        assert len(data) == len(self), f"self requires {len(self)} elements, but {key} has {len(data)} elements"
        self.edge_data[key] = data
    
    def pop_edata(self, key: str) -> Optional[TensorData]:
        return self.edge_data.pop(key, None)
    
    def pin_memory(self, device = None):
        state = {
            "node_data": {k:v.pin_memory(device=device) for k, v in self.node_data.items()},
            "edge_data": {k:v.pin_memory(device=device) for k, v in self.edge_data.items()},
        }

        if self.routes is None:
            state["routes"] = None
        else:
            state["routes"] = self.routes.pin_memory(device=device)
        
        for key, val in self.__dict__.items():
            if not isinstance(val, TensorData):
                continue
            state[key] = val.pin_memory(device=device)
        return type(self)(**state)
    
    def to(self, device = None, dtype = None, non_blocking = False, copy = False):
        state = {
            "node_data": {k:v.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy) for k, v in self.node_data.items()},
            "edge_data": {k:v.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy) for k, v in self.edge_data.items()},
        }

        if self.routes is None:
            state["routes"] = None
        else:
            state["routes"] = self.routes.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)

        for key, val in self.__dict__.items():
            if not isinstance(val, TensorData):
                continue
            state[key] = val.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        return type(self)(**state)

    def item(self, node_perm: Tensor | None = None, keep_ids = False):
        dataset = self.to_blocks(node_perm=node_perm, keep_ids=keep_ids)
        assert len(dataset) == 1, f"Expected 1 block, got {len(dataset)}"
        return dataset[0]

    def to_blocks(self, node_perm: Tensor | None = None, keep_ids = False):
        if node_perm is not None: # 创建映射表
            node_imap = torch.empty_like(node_perm)
            node_imap[node_perm] = torch.arange(
                node_perm.numel(),
                dtype=node_perm.dtype,
                device=node_perm.device,
            )

        dataset: List[DGLBlock] = []
        for i in range(len(self)):
            src = self.edge_src[i].item()
            dst = self.edge_dst[i].item()

            # 默认dst节点在src节点前面，self.src_ids只保存多出来的部分
            num_dst_nodes = self.dst_ids[i].data.numel()
            num_src_nodes = self.src_ids[i].data.numel() + num_dst_nodes

            if node_perm is not None:
                assert node_perm.numel() == num_dst_nodes
                dst = node_imap[dst]    # 直接映射
                
                m = src < num_dst_nodes # 只映射可行部分
                src[m] = node_imap[src[m]].type(src.dtype)
            
            g = dgl.create_block(
                (src, dst),
                num_src_nodes=num_src_nodes,
                num_dst_nodes=num_dst_nodes,
                idtype=torch.int32,
            )

            if keep_ids:
                dst_ids = self.dst_ids[i].item()
                if node_perm is not None:
                    dst_ids = dst_ids[node_perm] # 直接映射

                src_ids = torch.cat([
                    dst_ids,
                    self.src_ids[i].item(),
                ], dim=0)

                g.srcdata[dgl.NID] = src_ids
                g.dstdata[dgl.NID] = dst_ids
                g.edata[dgl.EID] = self.edge_ids[i].item()
            
            for key, val in self.node_data.items():
                val = val[i].item()
                if val.size(0) == num_src_nodes:
                    if node_perm is not None:
                        val = torch.cat([
                            val[node_perm],
                            val[num_dst_nodes:],
                        ], dim=0)

                    g.srcdata[key] = val
                elif val.size(0) == num_dst_nodes:
                    if node_perm is not None:
                        val = val[node_perm]
                        
                    g.dstdata[key] = val
                else:
                    raise ValueError(f"Node data {key} has invalid size {val.size(0)}")
            
            for key, val in self.edge_data.items():
                g.edata[key] = val[i].item()
            
            if self.routes is None:
                g.route = None
            else:
                g.route = self.routes[i].item()

                if node_perm is not None:
                    g.route.send_index = node_imap[g.route.send_index]

            dataset.append(g)
        return dataset

    @classmethod
    def from_blocks(cls, parts: List[DGLBlock]|DGLBlock):
        src_ids: List[Tensor] = []
        dst_ids: List[Tensor] = []
        
        edge_ids = []
        edge_src = []
        edge_dst = []

        routes = []
        src_node_data = {}
        dst_node_data = {}
        edge_data = {}
        node_data = {}
        if isinstance(parts,DGLBlock):
            parts = [parts]
        for g in parts:
            src, dst = g.edges()
            print('block src,dst size:', src.size(), dst.size())
            edge_src.append(src)
            edge_dst.append(dst)

            idtype = cast(Tensor, src).dtype
            device = cast(Tensor, dst).device

            if dgl.NID in g.srcdata:
                src_ids.append(g.srcdata[dgl.NID])
            else:
                src_ids.append(torch.arange(g.num_src_nodes(), dtype=idtype, device=device))

            if dgl.NID in g.dstdata:
                dst_ids.append(g.dstdata[dgl.NID])
            else:
                dst_ids.append(torch.arange(g.num_dst_nodes(), dtype=idtype, device=device))
            
            if dgl.EID in g.edata:
                edge_ids.append(g.edata[dgl.EID])
            else:
                edge_ids.append(torch.arange(g.num_edges(), dtype=idtype, device=device))
            
            # 从src_ids中剔除dst_ids前缀
            if src_ids[-1].numel() > dst_ids[-1].numel() and torch.any(src_ids[-1][:dst_ids[-1].numel()] != dst_ids[-1]):
                #raise ValueError("prefix of src_ids and dst_ids must be the same")
                src_ids[-1] = src_ids[-1][dst_ids[-1].numel():]

            routes.append(g.route) 

            for key in g.ndata['_N'].keys():
                if key not in node_data:
                    node_data[key] = []
                node_data[key].append(g.ndata['_N'][key])
            
            for key in g.srcdata.keys():
                if key == dgl.NID:
                    continue
                if key not in src_node_data:
                    src_node_data[key] = []
                src_node_data[key].append(g.srcdata[key])
            
            for key in g.dstdata.keys():
                if key == dgl.NID:
                    continue
                if key not in dst_node_data:
                    dst_node_data[key] = []
                dst_node_data[key].append(g.dstdata[key])
            

            for key in g.edata.keys():
                if key == dgl.EID:
                    continue
                if key not in edge_data:
                    edge_data[key] = []
                edge_data[key].append(g.edata[key])

        if len(dst_node_data) < len(src_node_data):
            node_data.update(dst_node_data)#{k:v for k,v in dst_node_data.items()}
            node_data.update(src_node_data)
        else:
            node_data.update(src_node_data)
            node_data.update(dst_node_data)
            
        src_ids = TensorData.from_tensors(src_ids)
        dst_ids = TensorData.from_tensors(dst_ids)

        edge_ids = TensorData.from_tensors(edge_ids)
        edge_src = TensorData.from_tensors(edge_src)
        edge_dst = TensorData.from_tensors(edge_dst)

        node_data = {k:TensorData.from_tensors(v) for k,v in node_data.items()}
        edge_data = {k:TensorData.from_tensors(v) for k,v in edge_data.items()}
        

        
        if None in routes:
            routes = None
        else:
            routes = RouteData.from_routes(routes)

        return cls(
            src_ids=src_ids,
            dst_ids=dst_ids,
            edge_ids=edge_ids,
            edge_src=edge_src,
            edge_dst=edge_dst,
            routes=routes,
            node_data=node_data,
            edge_data=edge_data,
        )
    
    def save(self, path):
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = {
            "node_data": {k:v.__getstate__() for k,v in self.node_data.items()},
            "edge_data": {k:v.__getstate__() for k,v in self.edge_data.items()},
        }

        if self.routes is None:
            state_dict["routes"] = None
        else:
            state_dict["routes"] = self.routes.__getstate__()

        for key, val in self.__dict__.items():
            if not isinstance(val, TensorData):
                continue
            state_dict[key] = val.__getstate__()
        torch.save(state_dict, str(path))
    
    @classmethod
    def load(cls, path, mmap: bool = False, shared_tensor: bool = True):
        path = Path(path).expanduser().resolve()
        
        state_dict = torch.load(str(path), mmap=mmap)

        state = {}
        for key, val in state_dict.items():
            if key == "routes":
                if val is None:
                    state[key] = None
                else:
                    state[key] = RouteData(**val)
            elif key == "node_data" or key == "edge_data":
                state[key] = {k:TensorData(**v) for k,v in val.items()}
            else:
                state[key] = TensorData(**val)
        return cls(**state)
    
    def set_device_type(self, device_type: str='c'):
        self.device_type = device_type