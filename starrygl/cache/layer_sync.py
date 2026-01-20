import torch
import torch.nn as nn
from typing import Optional, Tuple
from starrygl.cache.route import CacheRoute, PartitionState, CommPlan

class LayerSyncManager(nn.Module):
    """
    [通信逻辑独立封装]
    负责 GNN 层与层之间的 Embedding 同步。
    管理中间层显存 Cache，执行 Scatter -> Push -> Gather 流程。
    """
    def __init__(self, 
                 cache_router: CacheRoute, 
                 partition_state: PartitionState,
                 num_local_nodes: int, 
                 hidden_dim: int,
                 has_timestamp: bool = True):
        super().__init__()
        self.router = cache_router
        self.p_state = partition_state
        self.has_ts = has_timestamp
        
        # 中间层 Embedding 缓存 (GPU Buffer)
        # [Local_Size, Dim]
        self.register_buffer('emb_cache', torch.zeros(num_local_nodes, hidden_dim))
        if has_timestamp:
            self.register_buffer('ts_cache', torch.zeros(num_local_nodes))
            
        # 预定义回调函数 (避免在 forward 中重复定义)
        self._write_cb = self._update_cache_callback

    def _update_cache_callback(self, ids: torch.Tensor, feats: torch.Tensor, mode: str):
        """
        CacheRoute 接收到远程数据后的回调。
        ids: 本地 Cache 索引
        feats: 接收到的数据 [Batch, Dim + (1 if ts)]
        """
        if self.has_ts:
            # 假设最后一列是时间戳
            self.ts_cache[ids] = feats[:, 0]
            self.emb_cache[ids] = feats[:, 1:]
        else:
            self.emb_cache[ids] = feats

    def sync(self, 
             h_computed: torch.Tensor, 
             ts_computed: Optional[torch.Tensor],
             route: CommPlan, 
             out_gids: torch.Tensor, 
             next_in_gids: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        核心同步接口。
        
        流程:
        1. Scatter: 将当前 Batch 计算出的部分结果 (h_computed) 写入全量 Cache。
        2. Push: 根据 Route 将 Cache 中的数据发送给邻居 (Halo Update)。
        3. Gather: 从更新后的 Cache 中读取下一层所需的完整输入。
        
        Args:
            h_computed: 当前层计算出的 Embedding (对应 out_gids)
            ts_computed: 对应的时间戳 (可选)
            route: 通信计划
            out_gids: h_computed 对应的 Global ID (当前层的 dst nodes)
            next_in_gids: 下一层需要的 Input Global ID (下一层的 src nodes)
        """
        if route is None:
            # 如果没有路由（单机或无需同步），直接透传可能不对，因为 IDs 不对齐
            # 但通常如果有 route 为 None，说明不需要 Halo，意味着 next_in == out (除了维度)
            # 这里为了严谨，假设必须走 Cache 转换 ID
            pass

        # === 1. Scatter (Local Write) ===
        # 将 Batch 局部结果映射回 Global Cache 的本地位置
        local_out_idx = self.p_state.to_local_nid(out_gids)
        self.emb_cache[local_out_idx] = h_computed
        if self.has_ts and ts_computed is not None:
            self.ts_cache[local_out_idx] = ts_computed

        # === 2. Push (Remote Sync) ===
        # 准备发送的数据
        if self.has_ts:
            # 拼接 TS 和 Feat 以便一次发送
            # shape: [N, 1+D]
            local_data = torch.cat([self.ts_cache.unsqueeze(1), self.emb_cache], dim=1)
        else:
            local_data = self.emb_cache

        # 执行通信 (异步)
        futures = self.router.update_with_plan(
            local_features=local_data,
            plan=route,
            write_callback=self._write_cb
        )
        
        # 等待通信完成
        for f in futures: f.wait()

        # === 3. Gather (Read for Next Layer) ===
        # 下一层需要的所有节点 (包括刚刚从远端同步过来的 Halo)
        local_next_in_idx = self.p_state.to_local_nid(next_in_gids)
        
        h_next = self.emb_cache[local_next_in_idx]
        ts_next = self.ts_cache[local_next_in_idx] if self.has_ts else None
        
        return h_next, ts_next