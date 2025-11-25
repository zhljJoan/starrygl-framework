import torch

from torch import Tensor
from typing import *

from starrygl.core.route import Route


class RouteData:
    def __init__(self,
        send_sizes: List[List[int]],
        recv_sizes: List[List[int]],
        send_index_ind: Tensor | None,
        send_index_ptr: List[int] | None,
    ):
        assert len(send_sizes) == len(recv_sizes), "send_sizes and recv_sizes must have the same length"
        for i, (ss, rs) in enumerate(zip(send_sizes, recv_sizes)):
            assert len(ss) == len(rs), f"send_sizes[{i}] and recv_sizes[{i}] must have the same length"
        
        if send_index_ind is not None:
            assert send_index_ptr[-1] == send_index_ind.numel(), "send_index_ptr[-1] must equal send_index_ind.numel()"
        
        self.send_index_ind = send_index_ind
        self.send_index_ptr = send_index_ptr
        self.send_sizes = send_sizes
        self.recv_sizes = recv_sizes
    
    def __getstate__(self):
        return {
            "send_index_ind": self.send_index_ind,
            "send_index_ptr": self.send_index_ptr,
            "send_sizes": self.send_sizes,
            "recv_sizes": self.recv_sizes,
        }
    
    def __setstate__(self, state):
        self.send_index_ind = state["send_index_ind"]
        self.send_index_ptr = state["send_index_ptr"]
        self.send_sizes = state["send_sizes"]
        self.recv_sizes = state["recv_sizes"]
    
    def __repr__(self) -> str:
        n = len(self)
        p = len(self.send_sizes[0]) if self.send_sizes else -1
        return f"{type(self).__name__}(len={n}, parts={p})"
    
    def __len__(self):
        return len(self.send_sizes)
    
    def __getitem__(self, k: int | slice):
        if not isinstance(k, slice):
            k = slice(k, k + 1)

        if k.step is not None and k.step != 1:
            raise ValueError("Only step size of 1 is supported")
        s = 0 if k.start is None else k.start
        t = len(self) if k.stop is None else k.stop

        if self.send_index_ind is None:
            send_index_ind = self.send_index_ind
            send_index_ptr = self.send_index_ptr
        else:
            a, b = self.send_index_ptr[s], self.send_index_ptr[t]
            send_index_ind = self.send_index_ind[a:b]
            send_index_ptr = [x - a for x in self.send_index_ptr[s:t+1]]

        send_sizes = self.send_sizes[s:t]
        recv_sizes = self.recv_sizes[s:t]
        
        return type(self)(
            send_index_ind=send_index_ind,
            send_index_ptr=send_index_ptr,
            send_sizes=send_sizes,
            recv_sizes=recv_sizes,
        )
    
    def pin_memory(self, device = None):
        if self.send_index_ind is None:
            send_index_ind = self.send_index_ind
        else:
            send_index_ind = self.send_index_ind.pin_memory(device=device)

        return type(self)(
            send_index_ind=send_index_ind,
            send_index_ptr=self.send_index_ptr,
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
        )
    
    def to(self, device = None, dtype = None, non_blocking = False, copy = False):
        if self.send_index_ind is None:
            send_index_ind = self.send_index_ind
        else:
            send_index_ind = self.send_index_ind.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        return type(self)(
            send_index_ind=send_index_ind,
            send_index_ptr=self.send_index_ptr,
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
        )
    
    def item(self):
        routes = self.to_routes()
        assert len(routes) == 1, f"Expected 1 route, got {len(routes)}"
        return routes[0]
    
    def to_routes(self, group = None):
        routes: List[Route] = []
        for i in range(len(self)):
            if self.send_index_ind is None:
                send_index = None
            else:
                a, b = self.send_index_ptr[i], self.send_index_ptr[i+1]
                send_index = self.send_index_ind[a:b]

            r = Route(
                send_index=send_index,
                send_sizes=self.send_sizes[i],
                recv_sizes=self.recv_sizes[i],
                group=group,
            )
            routes.append(r)
        return routes
    
    @classmethod
    def from_routes(cls, routes: List[Route]):
        send_index_ind = []
        send_index_ptr = [0]
        send_sizes = []
        recv_sizes = []
        for r in routes:
            if r.send_index is None:
                assert not send_index_ind, "send_index_ind should be None if all routes have None send_index"
            else:
                send_index_ind.append(r.send_index)
                send_index_ptr.append(send_index_ptr[-1] + r.send_index.numel())

            send_sizes.append(r.send_sizes)
            recv_sizes.append(r.recv_sizes)
        
        if send_index_ind:
            assert len(send_index_ind) == len(routes), f"send_index_ind should have the same length as routes if any route has non-None send_index"
            send_index_ind = torch.cat(send_index_ind, dim=0)
        else:
            send_index_ind = None
            send_index_ptr = None

        return cls(
            send_index_ind=send_index_ind,
            send_index_ptr=send_index_ptr,
            send_sizes=send_sizes,
            recv_sizes=recv_sizes,
        )


class TensorData:
    def __init__(self, ptr: List[int], data: Tensor):
        assert ptr[-1] == data.size(0), f"ptr[-1] != data.size(0): {ptr[-1]} != {data.size(0)}"
        self.ptr = ptr
        self.data = data
    
    def __getstate__(self):
        return {
            "ptr": self.ptr,
            "data": self.data,
        }
    
    def __setstate__(self, state):
        self.ptr = state["ptr"]
        self.data = state["data"]
    
    def __repr__(self) -> str:
        n = len(self)
        s = tuple(self.data.size())
        return f"{type(self).__name__}(len={n}, size={s})"
    
    def __len__(self):
        return len(self.ptr) - 1
    
    def __getitem__(self, k: int | slice):
        if not isinstance(k, slice):
            k = slice(k, k + 1)

        if k.step is not None and k.step != 1:
            raise ValueError("Only step size of 1 is supported")
        s = 0 if k.start is None else k.start
        t = len(self) if k.stop is None else k.stop

        a, b = self.ptr[s], self.ptr[t]
        ptr = [x - a for x in self.ptr[s:t+1]]
        data = self.data[a:b]
        return type(self)(ptr=ptr, data=data)
    
    def pin_memory(self, device = None):
        data = self.data.pin_memory(device=device)
        return type(self)(ptr=self.ptr, data=data)
    
    def to(self, device = None, dtype = None, non_blocking = False, copy = False):
        data = self.data.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        return type(self)(ptr=self.ptr, data=data)
    
    def item(self):
        tensors = self.to_tensors()
        assert len(tensors) == 1, f"Expected 1 tensor, got {len(tensors)}"
        return tensors[0]
    
    def to_tensors(self):
        tensors: List[Tensor] = []
        for i in range(len(self)):
            a, b = self.ptr[i], self.ptr[i+1]
            tensors.append(self.data[a:b])
        return tensors
    
    @classmethod
    def from_tensors(cls, tensors: List[Tensor]):
        ptr = [0]
        for t in tensors:
            ptr.append(ptr[-1] + t.size(0))
        data = torch.cat(tensors, dim=0)
        return cls(ptr=ptr, data=data)

