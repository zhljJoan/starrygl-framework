import torch
import torch.distributed as dist
from typing import List, Optional, Tuple, Union
from collections import OrderedDict

class NodeState:
    """
    基础节点状态存储 (通常位于 CPU Pinned Memory)。
    只负责数据的存储、读取(get)和更新(update)，不负责通信。
    """
    def __init__(self, 
                 node_nums: int, 
                 dim: int, 
                 partition_state = None, # 保留参数兼容性
                 device: Union[str, torch.device] = 'cpu',
                 pinned: bool = True):
        
        self.node_nums = node_nums
        self.dim = dim
        self.device = torch.device(device)
        self.partition_state = partition_state
        
        # 核心数据存储
        self.data = torch.zeros(node_nums, dim, dtype=torch.float32, device=self.device)
        self.ts = torch.zeros(node_nums, dtype=torch.float32, device=self.device)
        
        # 启用 Pinned Memory (加速 CPU -> GPU 传输)
        if pinned and self.device.type == 'cpu':
            self.data = self.data.pin_memory()
            self.ts = self.ts.pin_memory()
            
    def get(self, node_ids: torch.Tensor, time_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        读取节点状态。
        Args:
            node_ids: 节点 ID (通常是 Global ID)
        Returns:
            (Embedding, Timestamp)
        """
        # 确保索引在同一设备
        if node_ids.device != self.device:
            node_ids = node_ids.to(self.device)
            
        # 切片读取
        emb = self.data[node_ids]
        ts = self.ts[node_ids]
        
        return emb, ts

    def update(self, node_ids: torch.Tensor, values: torch.Tensor, ts: Optional[torch.Tensor] = None):
        """
        更新节点状态。
        """
        if node_ids.device != self.device:
            node_ids = node_ids.to(self.device)
        if values.device != self.device:
            values = values.to(self.device)
            
        self.data[node_ids] = values
        
        if ts is not None:
            if ts.device != self.device:
                ts = ts.to(self.device)
            if ts.dim() > 1:
                ts = ts.squeeze()
            self.ts[node_ids] = ts

    # 兼容接口：Broadcast 现在等同于 Update (通信由外部 Router 控制)
    def broadcast(self, node_ids, values, ts):
        self.update(node_ids, values, ts)


class CachedNodeState(NodeState):
    """
    带有 GPU 缓存的节点状态。
    结合了 CPU 的全量存储和 GPU 的热点缓存 (Chunk-based)。
    """
    def __init__(
        self,
        data: NodeState,
        node_chunk_id: Optional[torch.Tensor] = None, 
        cache_capacity: int = 16, # 缓存多少个 Chunk
        device: torch.device = torch.device("cuda:0")
    ):
        # 初始化父类 (数据保留在原位置，通常是 CPU)
        super().__init__(data.node_nums, data.dim, data.partition_state, device=data.device, pinned=False)
        
        # 共享底层 CPU 数据 (引用)
        self.data = data.data
        self.ts = data.ts
        
        self.gpu_device = device
        self.feature_dim = data.dim
        
        # 如果没有提供 chunk_id (例如全图模式)，则禁用缓存逻辑，退化为普通 NodeState
        self.use_cache = (node_chunk_id is not None)
        
        if self.use_cache:
            self.node_chunk_id = node_chunk_id
            self.num_chunks = int(node_chunk_id.max().item()) + 1
            
            # 1. 预计算 Chunk 索引 (CPU -> GPU)
            print("[CachedNodeState] Building Cache Indices...")
            unique_chunks, counts = torch.unique(node_chunk_id, return_counts=True)
            self.max_chunk_size = int(counts.max().item())
            
            # 建立 node -> local_offset 映射
            # 为了简单，这里直接存储
            node_to_local = torch.zeros(self.node_nums, dtype=torch.long)
            self._chunk_to_nodes_cpu = []
            
            for c in range(self.num_chunks):
                mask = (node_chunk_id == c)
                nodes = torch.nonzero(mask, as_tuple=True)[0]
                self._chunk_to_nodes_cpu.append(nodes)
                # 计算在该 chunk 内的偏移
                node_to_local[nodes] = torch.arange(len(nodes))
                
            self.gpu_node_chunk_id = self.node_chunk_id.to(device)
            self.gpu_node_local_offset = node_to_local.to(device)
            
            # 2. GPU 缓存池 [Capacity, MaxSize, Dim+1] (+1 for timestamp)
            self.gpu_buffer = torch.zeros(
                (cache_capacity, self.max_chunk_size, self.feature_dim + 1),
                dtype=torch.float32, device=device
            )
            
            # 页表: chunk_id -> slot_id (-1 表示未缓存)
            self.gpu_page_table = torch.full((self.num_chunks,), -1, dtype=torch.long, device=device)
            self.chunk_to_slot_cpu = OrderedDict()
            self.free_slots = list(range(cache_capacity))
            
            self.prefetch_stream = torch.cuda.Stream(device=device)

    def get(self, node_ids: torch.Tensor, time_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据。
        如果启用了 Cache，先查 GPU，未命中则查 CPU。
        """
        if not self.use_cache:
            return super().get(node_ids, time_id)
            
        target_device = self.gpu_device
        if node_ids.device != self.gpu_device:
            node_ids = node_ids.to(self.gpu_device)
            
        # 查表
        chunk_ids = self.gpu_node_chunk_id[node_ids]
        slot_ids = self.gpu_page_table[chunk_ids]
        local_offsets = self.gpu_node_local_offset[node_ids]
        
        is_hit = (slot_ids != -1)
        
        # 结果容器
        out_combined = torch.empty((node_ids.size(0), self.feature_dim + 1), 
                                   device=self.gpu_device, dtype=torch.float32)
        
        # A. Hit 部分 (从 GPU Buffer 读)
        if is_hit.any():
            hit_idx = torch.nonzero(is_hit, as_tuple=True)[0]
            hit_slots = slot_ids[hit_idx]
            hit_offsets = local_offsets[hit_idx]
            out_combined[hit_idx] = self.gpu_buffer[hit_slots, hit_offsets]
            
        # B. Miss 部分 (从 CPU 读)
        if (~is_hit).any():
            miss_idx = torch.nonzero(~is_hit, as_tuple=True)[0]
            miss_nodes = node_ids[miss_idx]
            
            # 回退到 CPU 读取 (注意：super().get 在 CPU 上运行)
            cpu_emb, cpu_ts = super().get(miss_nodes.cpu())
            
            # 搬运到 GPU
            miss_combined = torch.cat([cpu_ts.unsqueeze(1), cpu_emb], dim=1)
            out_combined[miss_idx] = miss_combined.to(self.gpu_device)
            
        return out_combined[:, 1:], out_combined[:, 0]

    def update(self, node_ids: torch.Tensor, values: torch.Tensor, ts: Optional[torch.Tensor] = None):
        """
        同时更新 CPU 主存和 GPU 缓存。
        """
        # 1. 更新 CPU (Source of Truth)
        super().update(node_ids.cpu(), values.cpu(), ts.cpu() if ts is not None else None)
        
        # 2. 更新 GPU 缓存 (如果该节点在缓存中)
        if self.use_cache:
            if node_ids.device != self.gpu_device:
                node_ids = node_ids.to(self.gpu_device)
            if values.device != self.gpu_device:
                values = values.to(self.gpu_device)
            if ts is not None and ts.device != self.gpu_device:
                ts = ts.to(self.gpu_device)
                
            chunk_ids = self.gpu_node_chunk_id[node_ids]
            slot_ids = self.gpu_page_table[chunk_ids]
            
            mask = (slot_ids != -1)
            if mask.any():
                update_idx = torch.nonzero(mask, as_tuple=True)[0]
                
                u_slots = slot_ids[update_idx]
                u_offsets = self.gpu_node_local_offset[node_ids[update_idx]]
                
                u_vals = values[update_idx]
                u_ts = ts[update_idx] if ts is not None else torch.zeros_like(u_vals[:,0])
                
                combined = torch.cat([u_ts.unsqueeze(1), u_vals], dim=1)
                self.gpu_buffer[u_slots, u_offsets] = combined


class HistoryLayerUpdater:
    """
    辅助类：用于管理多层 History 的更新。
    """
    def __init__(self, node_states: List[NodeState]):
        self.node_states = node_states

    def update(self, layer_idx: int, nids: torch.Tensor, feats: torch.Tensor, ts: torch.Tensor):
        if layer_idx < len(self.node_states):
            self.node_states[layer_idx].update(nids, feats, ts)
            
    # 兼容旧接口
    def __call__(self, layer_idx, nids, feats, ts):
        self.update(layer_idx, nids, feats, ts)