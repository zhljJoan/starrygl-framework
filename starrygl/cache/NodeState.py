from collections import OrderedDict
from typing import Callable, Dict, List, Optional
import torch

from starrygl.cache.cache_route import CacheRoute
from starrygl.cache.replica_table import build_replica_table
from starrygl.route.route import PartitionState



class NodeState:
    def __init__(self, node_nums, dim, threshold = 0, threshold_time = 0, miss_times = 0, device = torch.device('cpu'), partition_state:PartitionState = None):
        
        self.data = torch.zeros(node_nums, dim, dtype=torch.float32, device=device)
        self.last_memory = torch.zeros(node_nums, dim, dtype=torch.float32, device=device)
        self.ts = torch.zeros(node_nums, dtype=torch.float32, device=device)
        self.last_time = torch.zeros(node_nums, dtype=torch.float32, device=device)
        self.increment_mem = torch.zeros(node_nums, dim, dtype=torch.float32, device=device)
        self.threshold = threshold
        self.threshold_time = threshold_time
        self.miss_times = miss_times
        self.node_nums = node_nums
        self.dim = dim
        self.update_count = torch.zeros(node_nums, dtype=torch.int64, device=device)
        self.device = device          
        self.last_update_info = None
        self.pending_futures = [] # 存储未完成的异步任务
        self.partition_state = partition_state  # 分区状态，用于分布式
        
    def get(self, node_id, time_id = 0, device: Optional[torch.device] = None, non_blocking: bool = True):
        return self.data[node_id].to(device=device, non_blocking=non_blocking), self.ts[node_id].to(device=device, non_blocking=non_blocking)
    
    
    def filter(self, index, value, ts):
        return torch.ones(index.size(0), dtype=torch.bool, device=self.device)
    
    #index是global id
    def broadcast(self, local_id, value, ts, filter:Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = None):
        """
        异步广播数据到所有节点。
        """
        if filter is not None:
            mask = filter(local_id, value, ts)
            local_id = local_id[mask]
            value = value[mask]
            ts = ts[mask]
        
        self.last_memory[local_id] = value
        self.last_time[local_id] = ts
        self.update_count[local_id] = 0
        global_id = self.partition_state.get_global_id(local_id)
        self.work = CacheRoute(torch.distributed.group.WORLD).update(
            features=torch.cat([ts.unsqueeze(1), value], dim=1),
            global_indices=global_id,
            replica_lookup_fn=self.partition_state.get_partition_book(),
            write_callback=self.update,
            is_hot_mask=self.partition_state.is_shared(local_id),
            slice_size=0  # 不切片，直接广播
        )
        self.pending_futures.append(self.work)
    
    def wait(self):
        for futures in self.pending_futures:
            for future in futures:
                future.wait()
        self.pending_futures.clear()
    
    def size(self, dim):
        if dim == 0: return self.node_nums
        if dim == 1: return self.dim
        return 0
    #node_ids是局部的
    def update(self, node_ids: torch.Tensor, ts: torch.Tensor, values: torch.Tensor, mode: str = 'write'):
        """
        更新主存中的节点状态。
        """
        if node_ids.device != self.device:
            node_ids = node_ids.to(self.device)
        if values.device != self.device:
            values = values.to(self.device)
        
        # 简单的覆盖更新 (根据具体算法可以是增量更新)
        self.data[node_ids] = values
        
        if ts is not None:
             if ts.device != self.device:
                ts = ts.to(self.device)
             self.ts[node_ids] = ts

    

class CachedNodeState(NodeState):
    
    def __init__(
        self,
        data: NodeState,
        node_chunk_id: torch.Tensor,  # (N,)
        cache_capacity: int = 16,
        device: torch.device = torch.device("cuda:0")
    ):
        # 保持数据在 CPU (最好是 pinned memory)
        super().__init__(data.node_nums, data.dim, data.threshold, data.threshold_time, data.miss_times, device=data.device, partition_state=data.partition_state)
        self.node_chunk_id = node_chunk_id
        self.device = device
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
        self.gpu_node_chunk_id = self.node_chunk_id.to(device, non_blocking=True)
        self.gpu_node_local_offset = node_to_local.to(device, non_blocking=True)

        # ==========================================
        # 2. 初始化 GPU 资源 (Cache Resources)
        # ==========================================
        # 显存池: [Capacity, Max_Chunk_Size, Feature_Dim]
        # 使用 3D Tensor 可以直接利用 [index, index] 高级索引
        self.gpu_buffer = torch.zeros(
            (cache_capacity, self.max_chunk_size, self.feature_dim + 1),
            dtype=self.data.dtype,
            device=device
        )
        
        # GPU 页表: 映射 ChunkID -> BufferSlotID
        # 初始化为 -1 表示未缓存
        self.gpu_page_table = torch.full((self.num_chunks,), -1, dtype=torch.long, device=device)
        
        # CPU 端的 LRU 管理 (维护映射关系)
        self.chunk_to_slot_cpu: OrderedDict[int, int] = OrderedDict()
        self.free_slots = list(range(cache_capacity))
        
        # 异步流
        self.prefetch_stream = torch.cuda.Stream(device=device)
        
    def prefetch_chunks(self, chunk_ids: List[int], time_id: int = 0):
        """
        异步预取 Chunks。
        这部分代码运行在 CPU 上，但发射 CUDA memcpy 命令到 prefetch_stream。
        """
        # 计算当前时间步的数据偏移
        #t_start = self.ptr[time_id]
        
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
                #src_indices = t_start + 
                src_indices = nodes_in_chunk
                
                # 2. 拷贝
                # 假设 self.data 是 pinned memory，这里的切片拷贝非常快
                src_tensor, src_ts = super().get(src_indices, time_id)
                src_tensor = torch.cat([src_ts.unsqueeze(1), src_tensor], dim=1)  # 添加时间戳
                # 写入 Buffer 的对应 Slot
                # 注意处理 chunk size < max_chunk_size 的情况
                real_size = src_tensor.size(0)
                self.gpu_buffer[slot_id, :real_size].copy_(src_tensor, non_blocking=True)

    def get(
        self,
        node_ids: torch.Tensor,
        time_id: int = 0,
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
            return out[:,1:], out[:,0]  # 返回特征和时间戳

        # === 场景 B: 部分命中或全未命中 (Hybrid Path) ===
        # 我们需要分别处理命中和未命中的部分，然后拼回去
        
        # 准备输出容器
        out = torch.empty((node_ids.size(0), self.feature_dim + 1), device=self.device, dtype=self.data.dtype)

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
            miss_nodes_cpu = miss_nodes.cpu()
            src_indices =  miss_nodes_cpu
            miss_features_cpu, miss_feature_ts = super().dataget(src_indices, time_id)
            
            # 4. 拷贝到 GPU 并填入输出 Tensor
            out[miss_indices] = torch.cat([miss_feature_ts.unsqueeze(1), miss_features_cpu], dim=1).to(self.device, non_blocking=non_blocking)

        # 如果用户请求的设备不是当前 GPU
        if target_device != self.device:
            out = out.to(target_device, non_blocking=non_blocking)
            
        return out[:, 1:], out[:, 0]  # 返回特征和时间戳
    
    def update(
        self,
        node_ids: torch.Tensor,
        ts: torch.Tensor,
        values: torch.Tensor,
        mode: str = 'write'
    ):
        """
        更新缓存中的节点状态。
        """
        # 1. 确保输入在正确的设备上
        if node_ids.device != self.device:
            node_ids = node_ids.to(self.device, non_blocking=True)
        if values.device != self.device:
            values = values.to(self.device, non_blocking=True)
        
        # 2. 更新主存
        super().update(node_ids, ts, values, mode)

        # 3. 更新 GPU 缓存
        chunk_ids = self.gpu_node_chunk_id[node_ids]
        slot_ids = self.gpu_page_table[chunk_ids]
        
        # 4. 找到需要更新的 slot
        update_mask = (slot_ids != -1)
        if update_mask.any():
            update_slots = slot_ids[update_mask]
            local_offsets = self.gpu_node_local_offset[node_ids[update_mask]]
            self.gpu_buffer[update_slots, local_offsets] = torch.cat([ts.unsqueeze(1), values], dim=1)
    # ================ 保持 TensorData 兼容性 ================
    
    def pin_memory(self, device=None):
        # 自身管理显存，不需要被 pin
        return self

    def to(self, device=None, **kwargs):
        # 这是一个 GPU 缓存类，to() 语义通常是指将结果转过去，
        # 或者本身就在 GPU 上。这里返回自身即可。
        return self

