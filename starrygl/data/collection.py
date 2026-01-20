import torch

from torch import Tensor
from typing import *

from starrygl.route.route import Route
from starrygl.utils.context import DistributedContext


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
    
    def __getitem__(self, k: int| slice):
        if not isinstance(k, slice):
            k = slice(k, k + 1)

        if k.step is not None and k.step != 1:
            raise ValueError("Only step size of 1 is supported")
        s = 0 if k.start is None else k.start
        t = len(self) if k.stop is None else k.stop
        print(s,t)
        a, b = self.ptr[s], self.ptr[t]
        ptr = [x - a for x in self.ptr[s:t+1]]
        data = self.data[a:b]
        return type(self)(ptr=ptr, data=data)
    
    def select(self, index, k: int|slice):
        if(len(self.ptr)==1):
            k = slice(0, 1) 
        if not isinstance(k, slice):
            k = slice(k, k + 1)

        if k.step is not None and k.step != 1:
            raise ValueError("Only step size of 1 is supported")
        s = 0 if k.start is None else k.start
        t = len(self) if k.stop is None else k.stop

        a, b = self.ptr[s], self.ptr[t]
        #ptr = [x - a for x in self.ptr[s:t+1]]
        len = index.shape[0]
        data = [self.data[i][index] for i in index]
        ptr = [0] + [data[i].shape for i in range(s, t+1)]
        data = torch.cat(data, dim=0)
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



class FastChunkCachedTensorData(TensorData):


    def __init__(
        self,
        ptr: List[int],
        data: torch.Tensor,
        node_chunk_id: torch.Tensor,  # (N,)
        cache_capacity: int = 16,
        device: torch.device = torch.device("cuda")
    ):
        # 保持数据在 CPU (最好是 pinned memory)
        super().__init__(ptr=ptr, data=data)
        
        self.node_chunk_id = node_chunk_id
        if device.type == 'cpu':
            pass
        else:
            ctx = DistributedContext.get_default_context()
            self.device = ctx.device
        self.cache_capacity = cache_capacity
        self.feature_dim = data.size(1)
        self.num_nodes = node_chunk_id.size(0)
        self.num_chunks = int(node_chunk_id.max().item()) + 1

        # ==========================================
        # 1. 预计算静态索引 (Static Indexing)
        # ==========================================
        # 计算每个 Chunk 的大小，以及每个节点在 Chunk 内的局部偏移
        # 这部分只在初始化做一次，计算完后全部转为 GPU Tensor 以便快速索引
        
        print("Pre-computing chunk indices...")
        # 统计每个 chunk 的大小，确定 Buffer 的形状 (Padding 到最大 chunk)
        unique_chunks, counts = torch.unique(node_chunk_id, return_counts=True)
        self.max_chunk_size = int(counts.max().item())
        
        # 计算 node_to_local_index (节点在 chunk 内的 0~k 偏移)
        # 使用 argsort 技巧快速计算
        sort_idx = torch.argsort(node_chunk_id) # 按 chunk 聚集节点
        sorted_chunk_ids = node_chunk_id[sort_idx]
        
        # 这里的逻辑是计算每个节点在排序后序列中相对于 chunk 开始位置的偏移
        # 既然 node_chunk_id 已经是 Tensor，我们可以利用 CPU 快速算完存起来
        node_to_local = torch.empty(self.num_nodes, dtype=torch.long)
        
        # 记录每个 chunk 包含的全局节点 ID (CPU list for prefetch slicing)
        self._chunk_to_global_nodes_cpu = []
        
        for c in range(self.num_chunks):
            # 找到属于 chunk c 的所有节点
            mask = (node_chunk_id == c)
            global_nodes = torch.nonzero(mask, as_tuple=True)[0]
            self._chunk_to_global_nodes_cpu.append(global_nodes)
            
            # 记录局部偏移
            node_to_local[global_nodes] = torch.arange(len(global_nodes))

        # 将静态映射表搬运到 GPU，用于 get_features 时的快速查表
        self.gpu_node_chunk_id = self.node_chunk_id.to(self.device, non_blocking=True)
        self.gpu_node_local_offset = node_to_local.to(self.device, non_blocking=True)

        # ==========================================
        # 2. 初始化 GPU 资源 (Cache Resources)
        # ==========================================
        # 显存池: [Capacity, Max_Chunk_Size, Feature_Dim]
        # 使用 3D Tensor 可以直接利用 [index, index] 高级索引
        self.gpu_buffer = torch.zeros(
            (cache_capacity, self.max_chunk_size, self.feature_dim),
            dtype=self.data.dtype,
            device=self.device
        )
        
        # GPU 页表: 映射 ChunkID -> BufferSlotID
        # 初始化为 -1 表示未缓存
        self.gpu_page_table = torch.full((self.num_chunks,), -1, dtype=torch.long, device=self.device)
        
        # CPU 端的 LRU 管理 (维护映射关系)
        self.chunk_to_slot_cpu: OrderedDict[int, int] = OrderedDict()
        self.free_slots = list(range(cache_capacity))
        
        # 异步流
        self.prefetch_stream = torch.cuda.Stream(device=self.device)

    def prefetch_chunks(self, chunk_ids: List[int], time_id: int):
        """
        异步预取 Chunks。
        这部分代码运行在 CPU 上，但发射 CUDA memcpy 命令到 prefetch_stream。
        """
        # 计算当前时间步的数据偏移
        t_start = self.ptr[time_id]
        
        with torch.cuda.stream(self.prefetch_stream):
            for cid in chunk_ids:
                if cid in self.chunk_to_slot_cpu:
                    self.chunk_to_slot_cpu.move_to_end(cid)
                    continue
                
                # --- 分配 Slot (LRU) ---
                if self.free_slots:
                    slot_id = self.free_slots.pop()
                else:
                    # 驱逐最旧的
                    evicted_cid, evicted_slot = self.chunk_to_slot_cpu.popitem(last=False)
                    # 在 GPU 页表中标记为无效 (实际上不标也没事，只要不访问)
                    # self.gpu_page_table[evicted_cid] = -1 
                    slot_id = evicted_slot
                
                # --- 更新元数据 ---
                self.chunk_to_slot_cpu[cid] = slot_id
                # 关键：更新 GPU 页表。这必须是一个 Tensor 操作，以便 get_features 可见。
                self.gpu_page_table[cid] = slot_id
                
                # --- 数据搬运 (H2D) ---
                # 1. 确定源数据位置
                nodes_in_chunk = self._chunk_to_global_nodes_cpu[cid]
                if nodes_in_chunk.numel() == 0:
                    continue
                    
                # 加上时间偏移
                src_indices = t_start + nodes_in_chunk
                
                # 2. 拷贝
                # 假设 self.data 是 pinned memory，这里的切片拷贝非常快
                src_tensor = self.data[src_indices] 
                
                # 写入 Buffer 的对应 Slot
                # 注意处理 chunk size < max_chunk_size 的情况
                real_size = src_tensor.size(0)
                self.gpu_buffer[slot_id, :real_size].copy_(src_tensor, non_blocking=True)

    def select(
        self,
        node_ids: torch.Tensor,
        time_id: int,
        device: Optional[torch.device] = None,
        non_blocking: bool = True
    ) -> torch.Tensor:
        """
        获取特征。
        自动处理缓存命中 (GPU Fast Path) 和缓存未命中 (CPU Slow Path)。
        """
        target_device = device or self.device
        
        # 确保 node_ids 在当前 GPU 设备上，以便查表
        if node_ids.device != self.device:
            node_ids = node_ids.to(self.device, non_blocking=True)

        # 1. 必须先等待预取流完成，防止读到写了一半的脏数据
        torch.cuda.current_stream().wait_stream(self.prefetch_stream)

        # 2. 查表 (全部在 GPU 上进行)
        chunk_ids = self.gpu_node_chunk_id[node_ids]      # Node -> Chunk
        slot_ids = self.gpu_page_table[chunk_ids]         # Chunk -> Slot
        local_offsets = self.gpu_node_local_offset[node_ids] # Node -> Local Offset

        # 3. 判断命中情况
        # slot_ids 为 -1 表示该 chunk 不在 GPU 缓存中
        is_hit = (slot_ids != -1)

        # === 场景 A: 全命中 (Fast Path, 最理想情况) ===
        if is_hit.all():
            # 直接使用高级索引提取
            out = self.gpu_buffer[slot_ids, local_offsets]
            if target_device != self.device:
                out = out.to(target_device, non_blocking=non_blocking)
            return out

        # === 场景 B: 部分命中或全未命中 (Hybrid Path) ===
        # 我们需要分别处理命中和未命中的部分，然后拼回去
        
        # 准备输出容器
        out = torch.empty((node_ids.size(0), self.feature_dim), device=self.device, dtype=self.data.dtype)

        # --- 处理命中部分 (GPU -> GPU) ---
        # 获取命中的索引位置
        hit_indices = torch.nonzero(is_hit, as_tuple=True)[0]
        if hit_indices.numel() > 0:
            hit_slots = slot_ids[hit_indices]
            hit_offsets = local_offsets[hit_indices]
            # 填入输出 Tensor
            out[hit_indices] = self.gpu_buffer[hit_slots, hit_offsets]

        # --- 处理未命中部分 (CPU -> GPU) ---
        miss_indices = torch.nonzero(~is_hit, as_tuple=True)[0]
        if miss_indices.numel() > 0:
            # 1. 获取未命中的 node_ids
            miss_nodes = node_ids[miss_indices]
            
            # 2. 计算在 CPU 原始 data 中的全局位置
            # 假设数据是按时间切片拼接的： ptr[time_id] 是当前时间步的起始位置
            # 如果 node_id 是 0~N 的全局 ID，且每个时间步都包含所有节点：
            global_offset_start = self.ptr[time_id]
            
            # 这里涉及一次 GPU -> CPU 的拷贝 (获取 miss_nodes 的值)
            # 因为要在 CPU 上做索引，这一步会引入同步开销，是 Cache Miss 的主要代价
            miss_nodes_cpu = miss_nodes.cpu()
            
            # 计算源地址
            src_indices = global_offset_start + miss_nodes_cpu
            
            # 3. 从 CPU 原始数据中读取 (Slow access)
            # 假设 self.data 是 Pinned Memory，这一步还算快
            miss_features_cpu = self.data[src_indices]
            
            # 4. 拷贝到 GPU 并填入输出 Tensor
            out[miss_indices] = miss_features_cpu.to(self.device, non_blocking=non_blocking)

        # 如果用户请求的设备不是当前 GPU
        if target_device != self.device:
            out = out.to(target_device, non_blocking=non_blocking)
            
        return out
    # ================ 保持 TensorData 兼容性 ================
    
    def pin_memory(self, device=None):
        # 自身管理显存，不需要被 pin
        return self

    def to(self, device=None, **kwargs):
        # 这是一个 GPU 缓存类，to() 语义通常是指将结果转过去，
        # 或者本身就在 GPU 上。这里返回自身即可。
        return self