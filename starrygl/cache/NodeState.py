import torch
from torch import Tensor
from typing import List, Optional, Union, Tuple

from starrygl.cache.cache_route import CacheRouteManager, CommPlan
from starrygl.utils.partition_book import DistRouteIndex, PartitionState

class DistNodeState:
    """
    高性能节点状态缓存 (Node Memory / Historical Embedding)。
    优化点：
    1. 整合 PartitionState，支持 Global ID/DistRouteIndex 访问。
    2. 使用 Pinned Memory 支持 GPU UVA 直接读取。
    3. 优化 Update 逻辑，确保异步写入时的设备对齐。
    """
    def __init__(self, 
                 state: PartitionState, 
                 dim: int, 
                 device: Union[str, torch.device] = 'cpu',
                 pinned: bool = True,
                 buffer_size = 1000000):
        self.state = state
        self.node_mapper = state.node_mapper
        self.loc_nums = self.node_mapper.loc_nums
        self.dim = dim
        self.device = torch.device(device)

        # 初始化存储：在指定设备上开辟空间
        # 默认放在 CPU Pinned Memory 以处理超大规模节点状态
        self.data = torch.zeros(self.loc_nums, dim, dtype=torch.float32, device=self.device)
        self.ts = torch.zeros(self.loc_nums, dtype=torch.float32, device=self.device)


        self.use_pinned = (pinned and self.device.type == 'cpu')
        if self.use_pinned:
            print('NodeState: Pinned Memory Enabled')
            self.data = self.data.pin_memory()
            self.ts = self.ts.pin_memory()
            
            # [新增] 预分配 Pinned Buffer，用于 gather 操作
            # 大小根据你的 max_batch_size * fanout 估算，不够时动态扩容
            self.buffer_size = buffer_size
            self.data_buffer = torch.zeros(buffer_size, dim, dtype=torch.float32).pin_memory()
            self.ts_buffer = torch.zeros(buffer_size, dtype=torch.float32).pin_memory()

    def __getitem__(self, key: Union[Tensor, DistRouteIndex]) -> Tuple[Tensor, Tensor]:
        """
        通过全局 ID 读取状态。支持 context.node_state[gids] 语法。
        """
        # 自动处理 ID 转换与设备对齐 (优化：优先使用 DistRouteIndex 快速路径)
        loc_idx = self.node_mapper.to_local(key, device=self.device)
        
        # UVA 读取：如果数据在 CPU Pinned，GPU 发起索引会直接走总线
        return self.data[loc_idx], self.ts[loc_idx]

    def update(self, gids: Tensor, values: Tensor, ts: Optional[Tensor] = None):
        """
        高性能异步更新。
        """
        loc_idx = self.node_mapper.to_local(gids, device=self.device)
        
        mask = loc_idx >= 0
        if not mask.any():
            return
            
        target_idx = loc_idx[mask]
        self.data[target_idx] = values[mask].to(self.device, non_blocking=True)
        
        if ts is not None:
            ts_to_update = ts[mask].to(self.device, self.ts.dtype,non_blocking=True)
            self.ts[target_idx] = ts_to_update.squeeze() if ts_to_update.dim() > 1 else ts_to_update

    def to(self, device: Union[str, torch.device]):
        """迁移状态存储设备 (例如从 CPU 整体迁移到 GPU)"""
        new_device = torch.device(device)
        if new_device != self.device:
            self.data = self.data.to(new_device)
            self.ts = self.ts.to(new_device)
            self.device = new_device
        return self

    def reset(self):
        """重置所有状态"""
        self.data.fill_(0)
        self.ts.fill_(0)

    def prefetch(self, gids: Tensor, stream: Optional[torch.cuda.Stream] = None, device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
        """
        Pinned Memory 优化版 Prefetch
        """
        if self.device.type == 'cpu' and self.use_pinned:
            loc_idx = self.state.node_mapper.to_local(gids, device=torch.device('cpu'))
            num_nodes = loc_idx.size(0)
            
            if num_nodes > self.data_buffer.size(0):
                new_size = int(num_nodes * 1.5)
                self.data_buffer = torch.zeros(new_size, self.dim, dtype=torch.float32).pin_memory()
                self.ts_buffer = torch.zeros(new_size, dtype=torch.float32).pin_memory()

            torch.index_select(self.data, 0, loc_idx, out=self.data_buffer[:num_nodes])
            torch.index_select(self.ts, 0, loc_idx, out=self.ts_buffer[:num_nodes])
            
            with torch.cuda.stream(stream):
                data_gpu = self.data_buffer[:num_nodes].to(device if device is not None else torch.device('cuda'), non_blocking=True)
                ts_gpu = self.ts_buffer[:num_nodes].to(device if device is not None else torch.device('cuda'), non_blocking=True)
                
            return data_gpu, ts_gpu
            
        elif self.device.type == 'cpu':
            with torch.cuda.stream(stream):
                loc_idx = self.state.node_mapper.to_local(gids, device=torch.device('cpu'))
                return self.data[loc_idx].to(device if device is not None else torch.device('cuda'), non_blocking=True), \
                          self.ts[loc_idx].to(device if device is not None else torch.device('cuda'), non_blocking=True)
        else:
            # 数据已经在 GPU 上，直接索引
            return self[gids]
    
    def reset(self):
        """重置所有状态"""
        self.data.fill_(0)
        self.ts.fill_(0)

    # 兼容接口：Broadcast 现在等同于 Update (通信由外部 Router 控制)
#    def broadcast(self, node_ids, values, ts):
#        self.update(node_ids, values, ts)


# class CachedNodeState(NodeState):
#     """
#     带有 GPU 缓存的节点状态。
#     结合了 CPU 的全量存储和 GPU 的热点缓存 (Chunk-based)。
#     """
#     def __init__(
#         self,
#         data: NodeState,
#         node_chunk_id: Optional[torch.Tensor] = None, 
#         cache_capacity: int = 16, # 缓存多少个 Chunk
#         device: torch.device = torch.device("cuda:0")
#     ):
#         # 初始化父类 (数据保留在原位置，通常是 CPU)
#         super().__init__(data.node_nums, data.dim, data.partition_state, device=data.device, pinned=False)
        
#         # 共享底层 CPU 数据 (引用)
#         self.data = data.data
#         self.ts = data.ts
        
#         self.gpu_device = device
#         self.feature_dim = data.dim
        
#         # 如果没有提供 chunk_id (例如全图模式)，则禁用缓存逻辑，退化为普通 NodeState
#         self.use_cache = (node_chunk_id is not None)
        
#         if self.use_cache:
#             self.node_chunk_id = node_chunk_id
#             self.num_chunks = int(node_chunk_id.max().item()) + 1
            
#             # 1. 预计算 Chunk 索引 (CPU -> GPU)
#             print("[CachedNodeState] Building Cache Indices...")
#             unique_chunks, counts = torch.unique(node_chunk_id, return_counts=True)
#             self.max_chunk_size = int(counts.max().item())
            
#             # 建立 node -> local_offset 映射
#             # 为了简单，这里直接存储
#             node_to_local = torch.zeros(self.node_nums, dtype=torch.long)
#             self._chunk_to_nodes_cpu = []
            
#             for c in range(self.num_chunks):
#                 mask = (node_chunk_id == c)
#                 nodes = torch.nonzero(mask, as_tuple=True)[0]
#                 self._chunk_to_nodes_cpu.append(nodes)
#                 # 计算在该 chunk 内的偏移
#                 node_to_local[nodes] = torch.arange(len(nodes))
                
#             self.gpu_node_chunk_id = self.node_chunk_id.to(device)
#             self.gpu_node_local_offset = node_to_local.to(device)
            
#             # 2. GPU 缓存池 [Capacity, MaxSize, Dim+1] (+1 for timestamp)
#             self.gpu_buffer = torch.zeros(
#                 (cache_capacity, self.max_chunk_size, self.feature_dim + 1),
#                 dtype=torch.float32, device=device
#             )
            
#             # 页表: chunk_id -> slot_id (-1 表示未缓存)
#             self.gpu_page_table = torch.full((self.num_chunks,), -1, dtype=torch.long, device=device)
#             self.chunk_to_slot_cpu = OrderedDict()
#             self.free_slots = list(range(cache_capacity))
            
#             self.prefetch_stream = torch.cuda.Stream(device=device)

#     def get(self, node_ids: torch.Tensor, time_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         获取数据。
#         如果启用了 Cache，先查 GPU，未命中则查 CPU。
#         """
#         if not self.use_cache:
#             return super().get(node_ids, time_id)
            
#         target_device = self.gpu_device
#         if node_ids.device != self.gpu_device:
#             node_ids = node_ids.to(self.gpu_device)
            
#         # 查表
#         chunk_ids = self.gpu_node_chunk_id[node_ids]
#         slot_ids = self.gpu_page_table[chunk_ids]
#         local_offsets = self.gpu_node_local_offset[node_ids]
        
#         is_hit = (slot_ids != -1)
        
#         # 结果容器
#         out_combined = torch.empty((node_ids.size(0), self.feature_dim + 1), 
#                                    device=self.gpu_device, dtype=torch.float32)
        
#         # A. Hit 部分 (从 GPU Buffer 读)
#         if is_hit.any():
#             hit_idx = torch.nonzero(is_hit, as_tuple=True)[0]
#             hit_slots = slot_ids[hit_idx]
#             hit_offsets = local_offsets[hit_idx]
#             out_combined[hit_idx] = self.gpu_buffer[hit_slots, hit_offsets]
            
#         # B. Miss 部分 (从 CPU 读)
#         if (~is_hit).any():
#             miss_idx = torch.nonzero(~is_hit, as_tuple=True)[0]
#             miss_nodes = node_ids[miss_idx]
            
#             # 回退到 CPU 读取 (注意：super().get 在 CPU 上运行)
#             cpu_emb, cpu_ts = super().get(miss_nodes.cpu())
            
#             # 搬运到 GPU
#             miss_combined = torch.cat([cpu_ts.unsqueeze(1), cpu_emb], dim=1)
#             out_combined[miss_idx] = miss_combined.to(self.gpu_device)
            
#         return out_combined[:, 1:], out_combined[:, 0]

#     def update(self, node_ids: torch.Tensor, values: torch.Tensor, ts: Optional[torch.Tensor] = None):
#         """
#         同时更新 CPU 主存和 GPU 缓存。
#         """
#         # 1. 更新 CPU (Source of Truth)
#         super().update(node_ids.cpu(), values.cpu(), ts.cpu() if ts is not None else None)
        
#         # 2. 更新 GPU 缓存 (如果该节点在缓存中)
#         if self.use_cache:
#             if node_ids.device != self.gpu_device:
#                 node_ids = node_ids.to(self.gpu_device)
#             if values.device != self.gpu_device:
#                 values = values.to(self.gpu_device)
#             if ts is not None and ts.device != self.gpu_device:
#                 ts = ts.to(self.gpu_device)
                
#             chunk_ids = self.gpu_node_chunk_id[node_ids]
#             slot_ids = self.gpu_page_table[chunk_ids]
            
#             mask = (slot_ids != -1)
#             if mask.any():
#                 update_idx = torch.nonzero(mask, as_tuple=True)[0]
                
#                 u_slots = slot_ids[update_idx]
#                 u_offsets = self.gpu_node_local_offset[node_ids[update_idx]]
                
#                 u_vals = values[update_idx]
#                 u_ts = ts[update_idx] if ts is not None else torch.zeros_like(u_vals[:,0])
                
#                 combined = torch.cat([u_ts.unsqueeze(1), u_vals], dim=1)
#                 self.gpu_buffer[u_slots, u_offsets] = combined


class HistoryLayerUpdater:
    """
    辅助类：用于管理多层 History 的更新。
    """
    
    def __init__(self, node_states: List[DistNodeState]):
        self.node_states = node_states
        self.task = None
    def update(self, layer_idx: int, nids: torch.Tensor, feats: torch.Tensor, ts: torch.Tensor):
        if layer_idx < len(self.node_states):
            self.node_states[layer_idx].update(nids, feats, ts)
            
    # 兼容旧接口
    def __call__(self, layer_idx, nids, feats, ts):
        self.update(layer_idx, nids, feats, ts)
    
    def reset(self):
        if self.task is not None:
            for t in self.task:
                t.wait()
        for state in self.node_states:
            state.reset()
            
    def update_embedding_and_broadcast(self, 
                                       layer_idx: int,
                                       updated_emb: torch.Tensor, 
                                       updated_ts: torch.Tensor, 
                                       plan: CommPlan,
                                       indices: torch.Tensor) -> List[torch.futures.Future]:
        """
        1. 更新本地存储 (Local Write)
        2. 根据通信计划，将更新异步发送给外部分区 (Global Broadcast)
        """
        assert layer_idx < len(self.node_states), "Layer index out of range"
        if self.task is not None:
            for t in self.task:
                t.wait()
            self.task = None
            
        if plan is None or plan.is_empty():
            return []

        if updated_ts.dim() == 1:
            updated_ts = updated_ts.unsqueeze(-1)
        
        #print(updated_emb.shape, updated_ts.shape, plan)
        combined_features = torch.cat([updated_emb, updated_ts], dim=-1)
        self.node_states[layer_idx].update(indices, updated_emb, updated_ts)
        def remote_write_callback(recv_indices, recv_data, mode):
            remote_emb = recv_data[:, :-1]
            remote_ts = recv_data[:, -1]
            target_state = self.node_states[layer_idx]
            target_state.data[recv_indices] = remote_emb.to(target_state.device, non_blocking=True)
            target_state.ts[recv_indices] = remote_ts.to(target_state.device, non_blocking=True)
        manager = CacheRouteManager.get_instance()
        futures = manager.update_with_plan(
            local_features=combined_features,
            plan=plan,
            write_callback=remote_write_callback
        )
        self.task = futures
        return futures
        