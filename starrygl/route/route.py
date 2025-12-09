import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.distributed as dist

import asyncio

from torch import Tensor
from typing import *

import dgl
from dgl.heterograph import DGLBlock

from starrygl.cache.replica_table import CSRReplicaTable, UVACSRReplicaTable

# from torch_scatter import scatter_add
from .timer import EventTimer

class DistRouteIndex:
    
    part_param = (0xFFFF ^ (1<<14))
    def __init__(self, index: Tensor, part_ids: Optional[Tensor] = None) -> None:
        if part_ids is None:
            self._data = index.long()
        else:
            index, part_ids = index.long(), part_ids.long()
            self._data = (index & 0xFFFFFFFFFFFF) | ((part_ids & 0xFFFF) << 48)
       
    @property
    def loc(self) -> Tensor:
        return self._data & 0xFFFFFFFFFFFF
    
    @property
    def part(self) -> Tensor:
        return (self._data >> 48) & self.part_param
    
    def set_shared(self, indx: slice):
        print(self._data.dtype, )
        self._data[indx] |= (1<<62)

    @property
    def is_shared(self):
        return (self._data>>62).to(torch.bool)
    
    @property
    def dist(self) -> Tensor:
        return self._data
    
    @property
    def dtype(self):
        return self._data.dtype
    
    @property
    def device(self):
        return self._data.device
    
    @property
    def shape(self):
        return self._data.shape
    
    def to(self,device) -> Tensor:
        return DistRouteIndex(self._data.to(device))
    
    def size(self,index):
        return self._data.size(index)

class PartitionState:
    def __init__(self, loc_ids: Tensor, loc_eids: Tensor,
                 is_shared: Tensor,
                 partition_book:CSRReplicaTable|UVACSRReplicaTable, 
                 dist_nid_mapper: Optional[Tensor] = None,
                 dist_eid_mapper: Optional[Tensor] = None):
        """
        partition_book: 分区映射，通常是一个字典或类似结构
        is_shared: 一个布尔张量，指示每个节点是否是共享的
        """
        self.partition_book = partition_book
        self.loc_ids = loc_ids
        self.loc_eids = loc_eids
        self.is_shared = is_shared
        self.dist_nid_mapper = dist_nid_mapper
        self.dist_eid_mapper = dist_eid_mapper
        
    def get_partition_book(self):
        return self.partition_book
    
    def is_shared(self, index: Tensor) -> Tensor:
        """
        返回一个布尔张量，指示哪些节点是共享的。
        """
        return self.is_shared[index]
    
    def get_global_id(self, local_id: Tensor) -> Tensor:
        """
        将本地 ID 转换为全局 ID。
        """
        return self.loc_ids[local_id]
    
    def get_global_eid(self, local_id: Tensor) -> Tensor:
        """
        将本地边 ID 转换为全局边 ID。
        """
        return self.loc_eids[local_id]
    
    def get_dist_id(self, local_id: Tensor) -> Tensor:
        """
        获取分布式 ID。
        """
        if self.dist_nid_mapper is not None:
            return self.dist_nid_mapper[self.loc_ids[local_id]]
    
    def get_dist_eid(self, local_id: Tensor) -> Tensor:
        """
        获取分布式边 ID。
        """
        if self.dist_eid_mapper is not None:
            return self.dist_eid_mapper[self.loc_eids[local_id]]
    
    
    
class Route:
    def __init__(self,
        send_sizes: List[int],
        recv_sizes: List[int],
        send_index: Tensor | None = None,
        group: dist.ProcessGroup | None = None,
        task = None
    ) -> None:
        self.send_sizes = send_sizes
        self.recv_sizes = recv_sizes
        self.send_index = send_index
        self.group = group
    
    @classmethod
    def get_timer(cls) -> EventTimer:
        if not hasattr(cls, "_timer_"):
            timer = EventTimer()
            setattr(cls, "_timer_", timer)
        return getattr(cls, "_timer_")
    
    @property
    def send_len(self) -> int:
        return sum(self.send_sizes)
    
    @property
    def recv_len(self) -> int:
        return sum(self.recv_sizes)
    
    def forward(self, x: Tensor, reverse: bool = False, group: dist.ProcessGroup | None = None) -> Tensor:
        group = self.group if group is None else group
        return RouteAgent(self, reverse=reverse, group=group).forward(x)
    
    async def async_forward(self, x: Tensor, reverse: bool = False, group: dist.ProcessGroup | None = None) -> Tensor:
        group = self.group if group is None else group
        return await RouteAgent(self, reverse=reverse, group=group).async_forward(x)
    
    def send(self, x: Tensor, reverse: bool = False, group: dist.ProcessGroup | None = None) -> Tensor:
        group = self.group if group is None else group
        return RouteAgent(self, reverse=reverse, group=group).send(x)
    
    def recv(self, ctx: Tensor) -> Tensor:
        return RouteAgent.recv(ctx)
    
    def pin_memory(self, device = None):
        return type(self)(
            index=self.send_index.pin_memory(device=device),
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
            group=self.group,
        )
    
    def to(self, device = None, dtype = None, non_blocking = False, copy = False):
        return type(self)(
            send_index=self.send_index.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
            group=self.group,
        )
    
    def __getstate__(self):
        return {
            "send_sizes": self.send_sizes,
            "recv_sizes": self.recv_sizes,
            "send_index": self.send_index,
        }
    
    def __setstate__(self, state):
        self.send_sizes = state["send_sizes"]
        self.recv_sizes = state["recv_sizes"]
        self.send_index = state["send_index"]
    
    def __repr__(self):
        p = len(self.send_sizes) if self.send_sizes else -1
        return f"{type(self).__name__}(parts={p})"
    
    @classmethod
    def from_empty(cls, num_nodes: int, edge_index: Tensor, idtype = torch.int32):
        src = edge_index[0]
        dst = edge_index[1]

        g = dgl.create_block(
            (src, dst),
            num_src_nodes=num_nodes,
            num_dst_nodes=num_nodes,
            idtype=idtype,
        )

        node_ids = torch.arange(num_nodes, dtype=idtype, device=edge_index.device)
        edge_ids = torch.arange(g.num_edges(), dtype=idtype, device=edge_index.device)

        g.srcdata[dgl.NID] = node_ids
        g.dstdata[dgl.NID] = node_ids
        g.edata[dgl.EID] = edge_ids

        # g.route = cls(
        #     send_sizes=[0],
        #     recv_sizes=[0],
        #     send_index=None,
        # )
        g.route = None
        return [g]
    
    @classmethod
    def all_to_all_ind2ptr(cls, dist_index: Union[Tensor, DistRouteIndex], group: Optional[dist.ProcessGroup] = None):
        """
        同步版本的 ind2ptr，返回 Route 对象和辅助索引。
        
        Returns:
            route: Route 对象，封装了通信计划
            recv_ind: 从其他 rank 接收到的索引
            sort_idx: 用于恢复原始顺序的排序索引
        """
        if isinstance(dist_index, Tensor):
            dist_index = DistRouteIndex(dist_index)
    
        device = dist_index.device
        world_size = dist.get_world_size(group)
        
        # 1. 排序
        sort_idx = torch.argsort(dist_index.part)
        sorted_loc = dist_index.loc[sort_idx]
        sorted_part = dist_index.part[sort_idx]
    
        # 2. 交换 Size
        send_counts = torch.bincount(sorted_part, minlength=world_size)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=group)
        
        send_counts_list = send_counts.tolist()
        recv_counts_list = recv_counts.tolist()
        total_recv_num = recv_counts.sum().item()
        
        # 3. 交换 Index
        recv_ind = torch.empty(total_recv_num, dtype=sorted_loc.dtype, device=device)
        dist.all_to_all_single(
            recv_ind, 
            sorted_loc, 
            output_split_sizes=recv_counts_list, 
            input_split_sizes=send_counts_list, 
            group=group
        )
        
        # 4. 构建 Route
        route = cls(
            send_sizes=send_counts_list,
            recv_sizes=recv_counts_list,
            send_index=sorted_loc,
            group=group
        )
        
        return route, recv_ind, sort_idx

    @classmethod
    async def all_to_all_ind2ptr_async(cls, dist_index: Union[Tensor, DistRouteIndex], group: Optional[dist.ProcessGroup] = None):
        """
        异步版本的 ind2ptr，返回 Route 对象、辅助索引以及 Work 句柄。
        
        Returns:
            route: Route 对象
            recv_ind: 接收缓冲区 (注意：在使用前必须等待 work 完成)
            sort_idx: 排序索引
            work: 异步通信句柄，调用者必须执行 `work.wait()`
        """
        if isinstance(dist_index, Tensor):
            dist_index = DistRouteIndex(dist_index)
    
        device = dist_index.device
        world_size = dist.get_world_size(group)
        
        # 1. 排序
        sort_idx = torch.argsort(dist_index.part)
        sorted_loc = dist_index.loc[sort_idx]
        sorted_part = dist_index.part[sort_idx]
    
        # 2. 交换 Size (这步通常很快，保持同步以简化逻辑)
        send_counts = torch.bincount(sorted_part, minlength=world_size)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=group)
        
        send_counts_list = send_counts.tolist()
        recv_counts_list = recv_counts.tolist()
        total_recv_num = recv_counts.sum().item()
        
        # 3. 启动异步交换 Index
        recv_ind = torch.empty(total_recv_num, dtype=sorted_loc.dtype, device=device)
        work = dist.all_to_all_single(
            recv_ind, 
            sorted_loc, 
            output_split_sizes=recv_counts_list, 
            input_split_sizes=send_counts_list, 
            group=group,
            async_op=True
        )
        
        # 让出 CPU
        await asyncio.sleep(0.0)
        
        # 4. 构建 Route
        route = cls(
            send_sizes=send_counts_list,
            recv_sizes=recv_counts_list,
            send_index=sorted_loc,
            group=group
        )
        
        return route, recv_ind, sort_idx, work
    @classmethod
    def from_subgraph(cls, node_parts:Tensor|int, is_shareds: Tensor, edge_index: Tensor, num_parts: int | None = None, idtype = torch.int32):
        pass
    
    @classmethod
    def from_graph(cls, node_parts: Tensor | int, edge_index: Tensor, num_parts: int | None = None, idtype = torch.int32):
        if not isinstance(node_parts, Tensor):
            num_nodes = int(node_parts)
            return cls.from_empty(num_nodes=num_nodes, edge_index=edge_index, idtype=idtype)
        
        if num_parts is None:
            num_parts = node_parts.max().item() + 1
        
        dst_ids_list: List[Tensor] = []
        edge_ids_list: List[Tensor] = []

        dst_send_ids_list: List[List[Tensor]] = []
        src_recv_ids_list: List[List[Tensor]] = []

        xmp = torch.empty_like(node_parts)
        for k in range(num_parts):
            dst_ids = torch.where(node_parts == k)[0]
            edge_ids = torch.where(node_parts[edge_index[1]] == k)[0]
            
            xmp.zero_()
            xmp[edge_index[0, edge_ids]] = 1
            xmp[dst_ids] = 0
            src_ids = torch.where(xmp != 0)[0]

            src_recv_ids: List[Tensor] = []
            for i in range(num_parts):
                src_recv_ids.append(src_ids[node_parts[src_ids] == i])
            
            dst_ids_list.append(dst_ids)
            edge_ids_list.append(edge_ids)
            src_recv_ids_list.append(src_recv_ids)
        
        for k in range(num_parts):
            dst_send_ids: List[Tensor] = []
            for i in range(num_parts):
                dst_send_ids.append(src_recv_ids_list[i][k])
            dst_send_ids_list.append(dst_send_ids)
        
        rets: List[DGLBlock] = []
        for k in range(num_parts):
            dst_ids = dst_ids_list[k]
            src_ids = torch.cat([
                dst_ids,
                torch.cat(src_recv_ids_list[k], dim=0),
            ], dim=0)

            xmp.fill_(2**63 - 1)
            xmp[src_ids] = torch.arange(src_ids.numel(), dtype=xmp.dtype, device=xmp.device)

            dst_send_ind = xmp[torch.cat(dst_send_ids_list[k], dim=0)]
            dst_send_szs = [t.numel() for t in dst_send_ids_list[k]]
            src_recv_szs = [t.numel() for t in src_recv_ids_list[k]]

            edge_ids = edge_ids_list[k]
            src = xmp[edge_index[0, edge_ids]]
            dst = xmp[edge_index[1, edge_ids]]

            g = dgl.create_block(
                (src, dst),
                num_src_nodes=src_ids.numel(),
                num_dst_nodes=dst_ids.numel(),
                idtype=idtype,
            )
            
            g.srcdata[dgl.NID] = src_ids.to(dtype=idtype)
            g.dstdata[dgl.NID] = dst_ids.to(dtype=idtype)
            g.edata[dgl.EID] = edge_ids.to(dtype=idtype)

            g.route = cls(
                send_sizes=dst_send_szs,
                recv_sizes=src_recv_szs,
                send_index=dst_send_ind,
            )
            rets.append(g)
        return rets
    
    def prune(self, dst_m: Tensor, group: dist.ProcessGroup | None = None):
        assert dst_m.dtype == torch.bool, f"dst_m must be a boolean tensor, but got {dst_m.dtype}"

        with torch.no_grad():
            src_m = self.forward(dst_m.long(), group=group)
            src_m = (src_m != 0)
        
        dst_ind = torch.arange(dst_m.numel(), dtype=torch.long, device=dst_m.device)
        src_ind = torch.arange(src_m.numel(), dtype=torch.long, device=src_m.device)

        send_index = []
        send_sizes = []

        recv_index = []
        recv_sizes = []

        dst_s = 0
        src_s = dst_m.numel()

        num_parts: int = len(self.send_sizes)
        for i in range(num_parts):
            x = self.send_index[dst_s:dst_s+self.send_sizes[i]]
            x = x[dst_m[x]]
            
            send_index.append(x)
            send_sizes.append(x.numel())

            x = src_ind[src_s:src_s+self.recv_sizes[i]]
            x = x[src_m[x]]

            recv_index.append(x)
            recv_sizes.append(x.numel())

            dst_s += self.send_sizes[i]
            src_s += self.recv_sizes[i]
        
        send_index = torch.cat(send_index, dim=0)
        new_src_ind = torch.cat([dst_ind] + recv_index, dim=0)
        
        route = type(self)(
            send_index=send_index,
            send_sizes=send_sizes,
            recv_sizes=recv_sizes,
        )

        route.group = self.group
        return route, src_m, new_src_ind


class RouteAgent:
    def __init__(self, route: Route, reverse: bool = False, group: dist.ProcessGroup | None = None) -> None:
        self.route = route
        self.group = group
        self.reverse = reverse
    
    def forward(self, x: Tensor) -> Tensor:
        if self.route.send_index is None:
            return x
        return self.recv(self.send(x))
    
    async def async_forward(self, x: Tensor) -> Tensor:
        if self.route.send_index is None:
            return x
        ctx = self.send(x)
        await asyncio.sleep(0.0)
        return self.recv(ctx)
    
    def send(self, x: Tensor) -> Tensor:
        if self.route.send_index is None:
            raise RuntimeError("Empty route is unsupported for send")
        return RouteSendFuncion.apply(x, self.route, self.reverse, self.group)
    
    @staticmethod
    def recv(ctx: Tensor) -> Tensor:
        return RouteRecvFunction.apply(ctx)


class RouteSendFuncion(autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, route: Route, reverse: bool = False, group: dist.ProcessGroup | None = None):
        r_ctx = RouteContext(route, reverse=reverse, group=group)
        r_ctx.forward_send(x)

        ret = torch.empty(0, dtype=torch.float32, device="cpu")
        ret._r_ctx = r_ctx

        ctx.saved_r_ctx = r_ctx
        return ret
    
    @staticmethod
    def backward(ctx, _):
        r_ctx: RouteContext = ctx.saved_r_ctx
        return r_ctx.backward_recv(), None, None, None
    

class RouteRecvFunction(autograd.Function):
    @staticmethod
    def forward(ctx, key: Tensor):
        r_ctx: RouteContext = key._r_ctx

        ctx.saved_r_ctx = r_ctx
        return r_ctx.forward_recv()
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        r_ctx: RouteContext = ctx.saved_r_ctx
        r_ctx.backward_send(grad_output)

        return torch.empty(0, dtype=torch.float32, device="cpu")



class RouteContext:
    def __init__(self, route: Route, reverse: bool = False, group: dist.ProcessGroup | None = None):
        self.route = route
        self.group = dist.GroupMember.WORLD if group is None else group
        self.reverse = reverse
        self.task: Union[Tensor, Tuple[Tensor, Tensor, dist.Work], None] = None

    def _send_impl(self, x: Tensor):
        y = x[self.route.send_index]
        outs = torch.empty(self.route.recv_len, *x.shape[1:], dtype=x.dtype, device=x.device)
        work = dist.all_to_all_single(
            outs, y,
            self.route.recv_sizes,
            self.route.send_sizes,
            group=self.group,
            async_op=True,
        )
        return x, outs, work

    def _send_post(self, x: Tensor, outs: Tensor, work: dist.Work):
        with Route.get_timer().record():
            work.wait()
        return torch.cat([x, outs], dim=0)

    def _recv_impl(self, x: Tensor):
        n = x.size(0) - self.route.recv_len
        x, y = x[:n], x[n:] # TODO: check if this is memory efficient
        outs = torch.empty(self.route.send_len, *x.shape[1:], dtype=x.dtype, device=x.device)
        work = dist.all_to_all_single(
            outs, y,
            self.route.send_sizes,
            self.route.recv_sizes,
            group=self.group,
            async_op=True,
        )
        return x, outs, work
    
    def _recv_post(self, x: Tensor, outs: Tensor, work: dist.Work):
        with Route.get_timer().record():
            work.wait()
        
        s = 0
        for sz in self.route.send_sizes:
            x[self.route.send_index[s:s+sz]] += outs[s:s+sz]
            s += sz
        # x = scatter_add(src=outs, index=self.route.send_index, dim=0, out=x)
        return x

    @torch.no_grad()
    def forward_send(self, x: Tensor):
        assert self.task is None
        x = x.detach()

        if self.reverse:
            self.task = self._recv_impl(x)
        else:
            self.task = self._send_impl(x)

    @torch.no_grad()
    def forward_recv(self):
        assert not isinstance(self.task, Tensor)

        if self.reverse:
            x = self._recv_post(*self.task)
        else:
            x = self._send_post(*self.task)

        self.task = None
        return x
    
    @torch.no_grad()
    def backward_send(self, g: Tensor):
        assert self.task is None
        g = g.detach()

        if self.reverse:
            self.task = self._send_impl(g)
        else:
            self.task = self._recv_impl(g)
    
    @torch.no_grad()
    def backward_recv(self):
        assert not isinstance(self.task, Tensor)

        if self.reverse:
            g = self._send_post(*self.task)
        else:
            g = self._recv_post(*self.task)
        
        self.task = None
        return g
    