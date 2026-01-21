import sys
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch import Tensor
from typing import Callable, Optional, Tuple, List, Union
from dataclasses import dataclass

from starrygl.cache.replica_table import CSRReplicaTable, UVACSRReplicaTable
from starrygl.route.route import DistRouteIndex

__all__ = ['PartitionState','CommPlan', 'CacheUpdateFuture', 'CacheRoute', 'CacheRouteManager']

# # =============================================================================
# # 1. CommPlan 数据定义 (原 Route)
# # =============================================================================

# class PartitionState:
#     """
#     维护当前分区的状态信息，包括：
#     1. 分布式元数据 (Partition Book, Shared Flags)
#     2. ID 映射服务 (Global ID <-> Local Index)
#     """
#     def __init__(self, loc_ids: Tensor, loc_eids: Tensor,
#                  is_shared: Tensor,
#                  partition_book: CSRReplicaTable | UVACSRReplicaTable, 
#                  dist_nid_mapper: Optional[Tensor] = None,
#                  dist_eid_mapper: Optional[Tensor] = None,
#                  device: Optional[torch.device] = None):
#         """
#         Args:
#             loc_ids (Tensor): 本地存储节点的 Global ID 列表。索引即为 Local ID。
#             loc_eids (Tensor): 本地存储边的 Global ID 列表。
#             is_shared (Tensor): 指示哪些节点是共享节点（Halo）。
#             partition_book: 分区表对象。
#             device: 映射表所在的设备，默认为 loc_ids.device。建议显式指定为 GPU。
#         """
#         self.partition_book = partition_book
#         self.loc_ids = loc_ids
#         self.loc_eids = loc_eids
#         self.is_shared_flag = is_shared
#         self.dist_nid_mapper = dist_nid_mapper
#         self.dist_eid_mapper = dist_eid_mapper
        
#         self.device = device if device is not None else loc_ids.device
        
#         # ID 映射表 (Global -> Local)
#         # 延迟初始化，或者在 init 时调用
#         self._nid_g2l_map: Optional[Tensor] = None
#         self._eid_g2l_map: Optional[Tensor] = None
        
#         # 自动构建映射 (如果是在 GPU 上，这一步会很快)
#         self.build_g2l_maps()

#     def build_g2l_maps(self):
#         """
#         构建 Global ID 到 Local Buffer Index 的映射表。
#         用于在 DataLoader 中将 batch 的全局 ID 转换为本地特征索引。
#         """
#         # --- Node Mapping ---
#         if self.loc_ids is not None:
#             max_nid = self.loc_ids.max().item()
#             # 使用 -1 表示该 Global ID 不在本地
#             self._nid_g2l_map = torch.full((max_nid + 1,), -1, dtype=torch.long, device=self.device)
            
#             # 填表: map[global_id] = local_index
#             ids = self.loc_ids.to(self.device)
#             self._nid_g2l_map[ids] = torch.arange(ids.size(0), dtype=torch.long, device=self.device)
            
#         # --- Edge Mapping ---
#         if self.loc_eids is not None:
#             max_eid = self.loc_eids.max().item()
#             self._eid_g2l_map = torch.full((max_eid + 1,), -1, dtype=torch.long, device=self.device)
            
#             eids = self.loc_eids.to(self.device)
#             self._eid_g2l_map[eids] = torch.arange(eids.size(0), dtype=torch.long, device=self.device)

#     def to_local_nid(self, global_ids: Tensor) -> Tensor:
#         """将全局节点 ID 转换为本地缓存索引"""
#         if self._nid_g2l_map is None:
#             raise RuntimeError("Node G2L map not initialized.")
#         return self._nid_g2l_map[global_ids.to(self.device)]

#     def to_local_eid(self, global_eids: Tensor) -> Tensor:
#         """将全局边 ID 转换为本地缓存索引 (用于读取 Edge Feature)"""
#         if self._eid_g2l_map is None:
#             raise RuntimeError("Edge G2L map not initialized (loc_eids is None).")
#         return self._eid_g2l_map[global_eids.to(self.device)]

#     def get_partition_book(self):
#         return self.partition_book
    
#     def is_shared(self, index: Tensor) -> Tensor:
#         return self.is_shared_flag[index]
    
#     def get_global_id(self, local_id: Tensor) -> Tensor:
#         return self.loc_ids[local_id]
    
#     def get_global_eid(self, local_id: Tensor) -> Tensor:
#         return self.loc_eids[local_id]
    
#     def get_dist_id(self, local_id: Tensor) -> Tensor:
#         if self.dist_nid_mapper is not None:
#             return self.dist_nid_mapper[self.loc_ids[local_id.to(self.loc_ids.device)]].to(local_id.device)
#         else:
#             return self.loc_ids[local_id]
    
#     def get_dist_eid(self, local_id: Tensor) -> Tensor:
#         if self.dist_eid_mapper is not None:
#             return self.dist_eid_mapper[self.loc_eids[local_id.to(self.loc_eids.device)]].to(local_id.device)
#         else:
#             return self.loc_eids[local_id]
    
#     def get_part(self, local_id: Tensor):
#         return DistRouteIndex(self.get_dist_id(local_id)).part
    
#     def get_epart(self, local_id:Tensor):
#         return DistRouteIndex(self.get_dist_eid(local_id)).part
    
@dataclass
class CommPlan:
    """
    通信计划 (Communication Plan)。
    替代原来的 Route 类。
    
    描述了在当前 Slot/Chunk 训练过程中，本地（Sender）需要向其他分区（Receiver）
    发送哪些节点的特征数据，以及对方应该写在什么位置。
    """
    send_ranks: torch.Tensor          # [Total_Send] 目标分区的 Rank ID 列表 (排序后的)
    send_sizes: torch.Tensor          # [Num_Parts] 发送给每个分区的节点数量
    send_indices: torch.Tensor        # [Total_Send] 本地读取索引 (Read Addr)
    send_remote_indices: torch.Tensor # [Total_Send] 远程写入索引 (Write Addr)

    def __repr__(self):
        if self.is_empty():
            return "CommPlan(Empty)"
        num_targets = torch.count_nonzero(self.send_sizes).item()
        total_items = self.send_indices.numel()
        return f"CommPlan(send={total_items} items -> {num_targets} parts)"

    def is_empty(self) -> bool:
        return self.send_indices.numel() == 0

    def to(self, device: torch.device, non_blocking: bool = False):
        self.send_ranks = self.send_ranks.to(device, non_blocking=non_blocking)
        self.send_sizes = self.send_sizes.to(device, non_blocking=non_blocking)
        self.send_indices = self.send_indices.to(device, non_blocking=non_blocking)
        self.send_remote_indices = self.send_remote_indices.to(device, non_blocking=non_blocking)
        return self

    def pin_memory(self):
        self.send_ranks = self.send_ranks.pin_memory()
        self.send_sizes = self.send_sizes.pin_memory()
        self.send_indices = self.send_indices.pin_memory()
        self.send_remote_indices = self.send_remote_indices.pin_memory()
        return self

    @classmethod
    def empty(cls, num_parts: int = 0):
        return cls(
            send_ranks=torch.empty(0, dtype=torch.long),
            send_sizes=torch.zeros(num_parts, dtype=torch.long) if num_parts > 0 else torch.empty(0, dtype=torch.long),
            send_indices=torch.empty(0, dtype=torch.long),
            send_remote_indices=torch.empty(0, dtype=torch.long)
        )

# =============================================================================
# 2. 异步句柄 (Future)
# =============================================================================

class CacheUpdateFuture:
    """
    异步更新句柄。调用 wait() 时才真正执行写入回调。
    """
    def __init__(
        self, 
        works: List[dist.Work], 
        recv_ids: Tensor,
        recv_feats: Tensor,
        write_callback: Callable[[Tensor, Tensor, str], None],
        mode: str
    ):
        self.works = works
        self.recv_ids = recv_ids
        self.recv_feats = recv_feats
        self.write_callback = write_callback
        self.mode = mode

    def wait(self):
        """同步等待通信完成并写入缓存"""
        for work in self.works:
            if work is not None:
                work.wait()
        
        # 执行写入
        if self.recv_ids.numel() > 0:
            self.write_callback(self.recv_ids, self.recv_feats, self.mode)

# =============================================================================
# 3. CacheRoute 通信核心
# =============================================================================

class CacheRoute:
    def __init__(self, group: Optional[dist.ProcessGroup] = None):
        self.group = group
        self.world_size = dist.get_world_size(group) if dist.is_initialized() else 1
        self.rank = dist.get_rank(group) if dist.is_initialized() else 0

    @torch.no_grad()
    def update_with_plan(
        self,
        local_features: Tensor,
        plan: CommPlan,  # 参数名改为 plan，类型改为 CommPlan
        write_callback: Callable[[Tensor, Tensor, str], None],
        slice_size: int = 0
    ) -> List[CacheUpdateFuture]:
        """
        [高性能模式] 使用预计算的 CommPlan 对象进行通信。
        """
        futures = []
        
        # 1. 检查是否为空
        if plan.is_empty():
            self._sync_empty_cold()
            return futures

        # 2. 交换大小 (Size Exchange)
        recv_counts = torch.empty_like(plan.send_sizes)
        dist.all_to_all_single(recv_counts, plan.send_sizes, group=self.group) # Sync
        
        send_list = plan.send_sizes.tolist()
        recv_list = recv_counts.tolist()
        total_recv = recv_counts.sum().item()

        # 3. 准备接收缓冲区
        recv_write_indices = torch.empty(total_recv, dtype=torch.long, device=local_features.device)
        recv_feats = torch.empty(total_recv, local_features.shape[1], dtype=local_features.dtype, device=local_features.device)

        works = []

        # 4. 发送写地址 (Remote Indices)
        work_idx = dist.all_to_all_single(
            recv_write_indices, plan.send_remote_indices,
            output_split_sizes=recv_list, input_split_sizes=send_list,
            group=self.group, async_op=True
        )
        works.append(work_idx)

        # 5. 发送特征 (Features)
        if slice_size <= 0 or slice_size >= local_features.shape[1]:
            packed_feats = local_features[plan.send_indices] # Gather
            work_f = dist.all_to_all_single(
                recv_feats, packed_feats,
                output_split_sizes=recv_list, input_split_sizes=send_list,
                group=self.group, async_op=True
            )
            works.append(work_f)
        
        else:
            D = local_features.shape[1]
            for start in range(0, D, slice_size):
                end = min(start + slice_size, D)
                chunk_feat = local_features[:, start:end]
                packed_chunk = chunk_feat[plan.send_indices]
                
                recv_chunk_view = recv_feats[:, start:end]
                work_slice = dist.all_to_all_single(
                    recv_chunk_view, packed_chunk,
                    output_split_sizes=recv_list, input_split_sizes=send_list,
                    group=self.group, async_op=True
                )
                works.append(work_slice)

        futures.append(CacheUpdateFuture(works, recv_write_indices, recv_feats, write_callback, 'cold'))
        return futures

    # @torch.no_grad()
    # def update(
    #     self,
    #     features: Tensor,
    #     global_indices: Tensor,
    #     replica_lookup_fn: Callable,
    #     write_callback: Callable,
    #     is_hot_mask: Optional[Tensor] = None,
    #     slice_size: int = 0
    # ) -> List[CacheUpdateFuture]:
    #     """
    #     [动态模式] 现场计算路由。
    #     """
    #     futures = []

    #     # 分离冷热
    #     if is_hot_mask is not None:
    #         cold_mask = ~is_hot_mask
    #         cold_feats = features[cold_mask]
    #         cold_ids = global_indices[cold_mask]
            
    #         hot_feats = features[is_hot_mask]
    #         hot_ids = global_indices[is_hot_mask]
            
    #         if hot_ids.numel() > 0:
    #             futures.append(self._async_broadcast_hot(hot_feats, hot_ids, write_callback))
    #     else:
    #         cold_feats = features
    #         cold_ids = global_indices

    #     # 动态冷数据分发
    #     if cold_ids.numel() > 0:
    #         futures.append(self._async_dispatch_cold_dynamic(
    #             cold_feats, cold_ids, replica_lookup_fn, write_callback, slice_size
    #         ))
    #     else:
    #         self._sync_empty_cold()

    #     return futures

    # def _async_dispatch_cold_dynamic(self, feats, ids, lookup_fn, callback, slice_size) -> CacheUpdateFuture:
    #     # 1. 动态查表
    #     src_indices, target_ranks, target_locs = lookup_fn(ids)
    #     if src_indices.numel() == 0:
    #         self._sync_empty_cold()
    #         return CacheUpdateFuture([], torch.empty(0), torch.empty(0), callback, 'cold')

    #     # 2. 排序
    #     sort_idx = torch.argsort(target_ranks)
    #     sorted_targets = target_ranks[sort_idx]
    #     sorted_locs = target_locs[sort_idx]
    #     src_indices_sorted = src_indices[sort_idx]

    #     # 3. 交换大小
    #     send_counts = torch.bincount(sorted_targets, minlength=self.world_size)
    #     recv_counts = torch.empty_like(send_counts)
    #     dist.all_to_all_single(recv_counts, send_counts, group=self.group)
        
    #     send_list = send_counts.tolist()
    #     recv_list = recv_counts.tolist()
    #     total_recv = recv_counts.sum().item()

    #     recv_ids = torch.empty(total_recv, dtype=ids.dtype, device=ids.device)
    #     recv_feats = torch.empty(total_recv, feats.shape[1], dtype=feats.dtype, device=feats.device)
    #     works = []

    #     # 4. 发送
    #     work_id = dist.all_to_all_single(
    #         recv_ids, sorted_locs, 
    #         output_split_sizes=recv_list, input_split_sizes=send_list, 
    #         group=self.group, async_op=True
    #     )
    #     works.append(work_id)

    #     if slice_size <= 0 or slice_size >= feats.shape[1]:
    #         expanded_feats = feats[src_indices_sorted]
    #         work_f = dist.all_to_all_single(
    #             recv_feats, expanded_feats,
    #             output_split_sizes=recv_list, input_split_sizes=send_list,
    #             group=self.group, async_op=True
    #         )
    #         works.append(work_f)
    #     else:
    #         D = feats.shape[1]
    #         for start in range(0, D, slice_size):
    #             end = min(start + slice_size, D)
    #             chunk_feat = feats[:, start:end]
    #             expanded_chunk = chunk_feat[src_indices_sorted]
    #             recv_chunk_view = recv_feats[:, start:end]
    #             work_slice = dist.all_to_all_single(
    #                 recv_chunk_view, expanded_chunk,
    #                 output_split_sizes=recv_list, input_split_sizes=send_list,
    #                 group=self.group, async_op=True
    #             )
    #             works.append(work_slice)

    #     return CacheUpdateFuture(works, recv_ids, recv_feats, callback, 'cold')

    # def _async_broadcast_hot(self, feats, ids, callback) -> CacheUpdateFuture:
    #     local_size = torch.tensor([feats.shape[0]], device=feats.device)
    #     all_sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
    #     dist.all_gather(all_sizes, local_size, group=self.group)
    #     sizes = [x.item() for x in all_sizes]
    #     max_size = max(sizes)
        
    #     if sum(sizes) == 0:
    #         return CacheUpdateFuture([], torch.empty(0), torch.empty(0), callback, 'hot')

    #     pad_len = max_size - feats.shape[0]
    #     if pad_len > 0:
    #         feat_in = torch.cat([feats, torch.zeros(pad_len, feats.shape[1], device=feats.device, dtype=feats.dtype)])
    #         id_in = torch.cat([ids, torch.zeros(pad_len, device=ids.device, dtype=ids.dtype)])
    #     else:
    #         feat_in, id_in = feats, ids

    #     feat_out = torch.empty(self.world_size * max_size, feats.shape[1], dtype=feats.dtype, device=feats.device)
    #     id_out = torch.empty(self.world_size * max_size, dtype=ids.dtype, device=ids.device)
        
    #     work_f = dist.all_gather_into_tensor(feat_out, feat_in, group=self.group, async_op=True)
    #     work_i = dist.all_gather_into_tensor(id_out, id_in, group=self.group, async_op=True)
        
    #     original_callback = callback
    #     def unpacking_callback(packed_ids, packed_feats, mode):
    #         valid_f, valid_i = [], []
    #         for i, s in enumerate(sizes):
    #             if s > 0:
    #                 start, end = i * max_size, i * max_size + s
    #                 valid_f.append(packed_feats[start:end])
    #                 valid_i.append(packed_ids[start:end])
    #         if valid_f:
    #             original_callback(torch.cat(valid_i), torch.cat(valid_f), mode)

    #     return CacheUpdateFuture([work_f, work_i], id_out, feat_out, unpacking_callback, 'hot')

    # def _sync_empty_cold(self):
    #     if not dist.is_initialized(): return
    #     rank = dist.get_rank(self.group)
    #     device = torch.device('cuda', rank)
    #     dummy = torch.tensor([], device=device)
    #     zeros = torch.zeros(self.world_size, dtype=torch.long, device=device)
    #     dist.all_to_all_single(zeros, zeros, group=self.group)
    #     dist.all_to_all_single(dummy, dummy, output_split_sizes=zeros.tolist(), input_split_sizes=zeros.tolist(), group=self.group)

# =============================================================================
# 4. 全局单例管理器
# =============================================================================

class CacheRouteManager:
    _instance = None
    @classmethod
    def get_instance(cls, group: Optional[dist.ProcessGroup] = None) -> CacheRoute:
        if cls._instance is None:
            cls._instance = CacheRoute(group)
        return cls._instance