import torch
from torch import Tensor
from typing import List, Optional, Union, Tuple

from starrygl.cache.cache_route import CacheRouteManager, CommPlan
from starrygl.utils.partition_book import DistRouteIndex, PartitionState

import torch
from torch import Tensor
from typing import Optional, Union, Tuple

class DistNodeState:
    def __init__(self, 
                 state, 
                 dim: int, 
                 device: Union[str, torch.device] = 'cpu',
                 pinned: bool = True,
                 buffer_size: int = 1000000):
        self.state = state
        self.node_mapper = state.node_mapper
        self.loc_nums = self.node_mapper.loc_nums
        self.dim = dim
        self.device = torch.device(device)

        # 主存储空间
        self.data = torch.zeros(self.loc_nums, dim, dtype=torch.float32, device=self.device)
        self.ts = torch.zeros(self.loc_nums, dtype=torch.float32, device=self.device)

        self.use_pinned = (pinned and self.device.type == 'cpu')
        if self.use_pinned:
            # 确保主存储是 Pinned 状态
            self.data = self.data.pin_memory()
            self.ts = self.ts.pin_memory()
            
            # [核心优化] 预分配足够大的 Pinned Buffer 用于异步更新
            # 避免在 update 时临时分配导致 cudaHostAlloc 阻塞
            self.data_buffer = torch.zeros(buffer_size, dim, dtype=torch.float32).pin_memory()
            self.ts_buffer = torch.zeros(buffer_size, dtype=torch.float32).pin_memory()
            self._buffer_ptr = 0

    def update(self, gids: Tensor, values: Tensor, ts: Optional[Tensor] = None):
        """
        高性能异步更新：强制走 Pinned -> GPU 路径，消除发射气泡。
        """
        # 1. 局部 ID 转换 (在 CPU 上进行)
        loc_idx = self.node_mapper.to_local(gids, device=torch.device('cpu'))
        mask = loc_idx >= 0
        if not mask.any():
            return
            
        target_idx = loc_idx[mask]
        num_updates = target_idx.size(0)

        # 2. 动态扩容检查 (仅在极其必要时触发)
        if num_updates > self.data_buffer.size(0):
            new_size = int(num_updates * 1.2)
            self.data_buffer = torch.zeros(new_size, self.dim, dtype=torch.float32).pin_memory()
            self.ts_buffer = torch.zeros(new_size, dtype=torch.float32).pin_memory()

        # 3. [关键步骤] 先拷贝到 Pinned Buffer
        # 这样 values (Pageable) 到 data_buffer (Pinned) 的 copy 是由 CPU 完成的
        self.data_buffer[:num_updates].copy_(values[mask])
        
        # 4. 真正的异步搬运到 GPU
        # 因为 data_buffer 是 pinned，non_blocking=True 才会生效
        self.data[target_idx] = self.data_buffer[:num_updates].to(self.device, non_blocking=True)
        
        if ts is not None:
            self.ts_buffer[:num_updates].copy_(ts[mask].squeeze())
            self.ts[target_idx] = self.ts_buffer[:num_updates].to(self.device, non_blocking=True)

    def prefetch(self, gids: Tensor, stream: Optional[torch.cuda.Stream] = None, device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
        """
        利用 UVA 优化读取
        """
        loc_idx = self.state.node_mapper.to_local(gids, device=torch.device('cpu'))
        num_nodes = loc_idx.size(0)
        
        # 直接使用 index_select 到预分配的 pinned 容器
        torch.index_select(self.data, 0, loc_idx, out=self.data_buffer[:num_nodes])
        torch.index_select(self.ts, 0, loc_idx, out=self.ts_buffer[:num_nodes])
        
        with torch.cuda.stream(stream):
            # 这里的 to 也是非阻塞的
            data_gpu = self.data_buffer[:num_nodes].to(device or torch.device('cuda'), non_blocking=True)
            ts_gpu = self.ts_buffer[:num_nodes].to(device or torch.device('cuda'), non_blocking=True)
                
        return data_gpu, ts_gpu
    
    def reset(self):
        self.data.zero_()
        self.ts.zero_()

class HistoryLayerUpdater:
    def __init__(self, node_states: List[DistNodeState]):
        self.node_states = node_states
        self.task_queue = [] # 改为队列管理，支持流水线

    def update_embedding_and_broadcast(self, 
                                       layer_idx: int,
                                       updated_emb: torch.Tensor, 
                                       updated_ts: torch.Tensor, 
                                       plan: CommPlan,
                                       indices: torch.Tensor):
        """
        全异步广播优化。
        """
        assert layer_idx < len(self.node_states), "Layer index out of range"
        
        # [优化] 移除硬同步 wait()
        # 仅在队列过长时清理，防止 OOM
        if len(self.task_queue) > 2:
            old_tasks = self.task_queue.pop(0)
            for t in old_tasks: t.wait()
            
        if plan is None or plan.is_empty():
            # 虽然没广播，但本地状态仍需异步更新
            self.node_states[layer_idx].update(indices, updated_emb, updated_ts)
            return []

        # [性能点] 避免频繁 cat，改用预分配的视图（如果可能）
        if updated_ts.dim() == 1:
            updated_ts = updated_ts.unsqueeze(-1)
        
        # 仅在必要时才进行 cat 操作（NCCL 发送需要连续空间）
        combined_features = torch.cat([updated_emb, updated_ts], dim=-1).contiguous()
        
        # 1. 发起本地异步更新
        self.node_states[layer_idx].update(indices, updated_emb, updated_ts)

        # 2. 定义远程写入回调 (确保回调内部也是 non_blocking)
        def remote_write_callback(recv_indices, recv_data, mode):
            # recv_data 已经在 GPU 上
            remote_emb = recv_data[:, :-1]
            remote_ts = recv_data[:, -1]
            target_state = self.node_states[layer_idx]
            
            # 远程数据的写入也应确保不阻塞主计算流
            target_state.data[recv_indices] = remote_emb.to(target_state.device, non_blocking=True)
            target_state.ts[recv_indices] = remote_ts.to(target_state.device, non_blocking=True)

        # 3. 发起分布式异步广播 ( manager 内部应使用独立 stream)
        manager = CacheRouteManager.get_instance()
        futures = manager.update_with_plan(
            local_features=combined_features,
            plan=plan,
            write_callback=remote_write_callback
        )
        
        self.task_queue.append(futures)
        return futures

    def reset(self):
        # 重置时才进行最终同步
        for tasks in self.task_queue:
            for t in tasks: t.wait()
        self.task_queue = []
        for state in self.node_states:
            state.reset()