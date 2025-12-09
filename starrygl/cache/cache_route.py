import torch
import torch.distributed as dist
from torch import Tensor
from typing import Callable, Optional, Tuple, List, Union

class CacheUpdateFuture:
    """
    异步更新的句柄 (Future/Promise)。
    持有通信的 Work 句柄和接收缓冲区，调用 wait() 时才真正执行写入。
    """
    def __init__(
        self, 
        works: List[dist.Work], 
        recv_ids: Tensor,
        recv_feats: Tensor,
        write_callback: Callable,
        mode: str
    ):
        self.works = works
        self.recv_ids = recv_ids
        self.recv_feats = recv_feats
        self.write_callback = write_callback
        self.mode = mode

    def wait(self):
        """同步点：等待通信完成并写入缓存"""
        # 1. 等待所有 NCCL 任务完成
        for work in self.works:
            if work is not None:
                work.wait()
        
        # 2. 执行回调写入本地缓存
        if self.recv_ids.numel() > 0:
            self.write_callback(self.recv_ids, self.recv_feats[0], self.recv_feats[1:], self.mode)


class CacheRoute:
    def __init__(self, group: Optional[dist.ProcessGroup] = None):
        self.group = group
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        
        

    @torch.no_grad()
    def update(
        self,
        features: Tensor,
        global_indices: Tensor,
        replica_lookup_fn: Callable[[Tensor], Tuple[Tensor, Tensor]],
        write_callback: Callable[[Tensor, Tensor, str], None],
        is_hot_mask: Optional[Tensor] = None,
        slice_size: int = 0  # 0 表示不切片（追求最快），设置具体数值（如 32）可节省显存
    ) -> List[CacheUpdateFuture]:
        """
        启动异步更新。返回 Future 列表（可能包含冷/热两个任务）。
        
        Args:
            slice_size: 如果显存紧张（避免复制多份导致OOM），设置此值（例如 32 或 64）。
                        会开启流水线：一边发 Part A，一边准备 Part B。
        """
        futures = []

        # --- 1. 数据分流 ---
        if is_hot_mask is not None:
            cold_mask = ~is_hot_mask
            # 注意：这里的 mask 索引会触发 copy，无法完全 zero-copy，
            # 但这是分离冷热数据的必经之路。
            cold_feats = features[cold_mask]
            cold_ids = global_indices[cold_mask]
            
            # 热点数据通常较少，直接走广播，不切片
            hot_feats = features[is_hot_mask]
            hot_ids = global_indices[is_hot_mask]
            
            if hot_ids.numel() > 0:
                futures.append(self._async_broadcast_hot(hot_feats, hot_ids, write_callback))
        else:
            cold_feats = features
            cold_ids = global_indices

        # --- 2. 启动冷数据更新 ---
        if cold_ids.numel() > 0:
            futures.append(self._async_dispatch_cold(
                cold_feats, cold_ids, replica_lookup_fn, write_callback, slice_size
            ))
        else:
            # 防止死锁：本地没数据也要参与同步
            self._sync_empty_cold()

        return futures

    def _async_dispatch_cold(
        self, 
        feats: Tensor, 
        ids: Tensor, 
        lookup_fn: Callable, 
        callback: Callable,
        slice_size: int
    ) -> CacheUpdateFuture:
        
        # A. 查找副本归属 (1-to-Many)
        src_indices, target_ranks, target_locs = lookup_fn(ids)
        if src_indices.numel() == 0:
            self._sync_empty_cold()
            return CacheUpdateFuture([], torch.empty(0), torch.empty(0), callback, 'cold')

        # B. 排序与元数据交换 (这部分必须同步，很快)
        sort_idx = torch.argsort(target_ranks)
        sorted_targets = target_ranks[sort_idx]
        sorted_locs = target_locs[sort_idx]
        # 只需重排一次 ID


        send_counts = torch.bincount(sorted_targets, minlength=self.world_size)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group) # Sync
        
        send_list = send_counts.tolist()
        recv_list = recv_counts.tolist()
        total_recv = recv_counts.sum().item()

        # C. 准备接收缓冲区
        recv_ids = torch.empty(total_recv, dtype=ids.dtype, device=ids.device)
        recv_feats = torch.empty(total_recv, feats.shape[1], dtype=feats.dtype, device=feats.device)

        works = []

        # D. 发送 ID (Async)
        work_id = dist.all_to_all_single(
            recv_ids, sorted_locs, 
            output_split_sizes=recv_list, input_split_sizes=send_list, 
            group=self.group, async_op=True
        )
        works.append(work_id)

        # E. 发送特征 (核心优化：Slice Pipeline vs Single Shot)
        # src_indices_sorted 只是索引，还没发生特征数据的物理复制
        src_indices_sorted = src_indices[sort_idx] 

        if slice_size <= 0 or slice_size >= feats.shape[1]:
            # === 模式 1: 极速模式 (显存够用) ===
            # 直接复制所有特征，一次发送
            expanded_feats = feats[src_indices_sorted] # 显存复制发生在这里
            work_f = dist.all_to_all_single(
                recv_feats, expanded_feats,
                output_split_sizes=recv_list, input_split_sizes=send_list,
                group=self.group, async_op=True
            )
            works.append(work_f)
        else:
            # === 模式 2: 省显存流水线 (Slicing) ===
            # 将特征切块，循环发送。
            # 这样显存里只需要存一个 Slice 的副本，而不是 3x 整个特征。
            D = feats.shape[1]
            for start in range(0, D, slice_size):
                end = min(start + slice_size, D)
                
                # 1. 切片 + 复制 (只复制这一小块)
                # 注意：这里会产生临时的显存占用，但用完即弃
                chunk_feat = feats[:, start:end]
                expanded_chunk = chunk_feat[src_indices_sorted] # Expansion
                
                # 2. 发送 (Async)
                # 直接 view 到接收的大 buffer 里
                recv_chunk_view = recv_feats[:, start:end]
                
                work_slice = dist.all_to_all_single(
                    recv_chunk_view, expanded_chunk,
                    output_split_sizes=recv_list, input_split_sizes=send_list,
                    group=self.group, async_op=True
                )
                works.append(work_slice)
                
                # 关键：这里没有 wait()！
                # 当 CPU 进入下一次循环准备下一个 Slice 时，GPU 正在发送上一个 Slice。
                # 完美流水线。

        return CacheUpdateFuture(works, recv_ids, recv_feats, callback, 'cold')

    def _async_broadcast_hot(self, feats, ids, callback) -> CacheUpdateFuture:
        # 1. Exchange Sizes (Sync)
        local_size = torch.tensor([feats.shape[0]], device=feats.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        dist.all_gather(all_sizes, local_size, group=self.group)
        sizes = [x.item() for x in all_sizes]
        max_size = max(sizes)
        
        if sum(sizes) == 0:
            return CacheUpdateFuture([], torch.empty(0), torch.empty(0), callback, 'hot')

        # 2. Padding
        pad_len = max_size - feats.shape[0]
        if pad_len > 0:
            feat_in = torch.cat([feats, torch.zeros(pad_len, feats.shape[1], device=feats.device, dtype=feats.dtype)])
            id_in = torch.cat([ids, torch.zeros(pad_len, device=ids.device, dtype=ids.dtype)])
        else:
            feat_in, id_in = feats, ids

        # 3. Async Gather
        feat_out = torch.empty(self.world_size * max_size, feats.shape[1], dtype=feats.dtype, device=feats.device)
        id_out = torch.empty(self.world_size * max_size, dtype=ids.dtype, device=ids.device)
        
        work_f = dist.all_gather_into_tensor(feat_out, feat_in, group=self.group, async_op=True)
        work_i = dist.all_gather_into_tensor(id_out, id_in, group=self.group, async_op=True)
        
        # 4. 后处理回调闭包 (Handle Padding/Unpacking later)
        # 我们需要在 Future.wait() 里做解包，所以 wrap 一个稍微复杂的 callback
        original_callback = callback
        
        def unpacking_callback(packed_ids, packed_feats, mode):
            # 解包逻辑
            valid_f, valid_i = [], []
            for i, s in enumerate(sizes):
                if s > 0:
                    start, end = i * max_size, i * max_size + s
                    valid_f.append(packed_feats[start:end])
                    valid_i.append(packed_ids[start:end])
            if valid_f:
                original_callback(torch.cat(valid_i), torch.cat(valid_f), mode)

        return CacheUpdateFuture([work_f, work_i], id_out, feat_out, unpacking_callback, 'hot')

    def _sync_empty_cold(self):
        """空数据同步，防止 NCCL 死锁"""
        dummy = torch.tensor([], device=torch.device('cuda', self.rank))
        zeros = torch.zeros(self.world_size, dtype=torch.long, device=dummy.device)
        # 必须同步，不能 async，否则这里的临时变量销毁可能导致错误
        dist.all_to_all_single(zeros, zeros, group=self.group)
        dist.all_to_all_single(dummy, dummy, output_split_sizes=zeros.tolist(), input_split_sizes=zeros.tolist(), group=self.group)
        
class CacheRouteManager:
    """
    管理 CacheRoute 实例，提供全局访问。
    """
    _instance = None

    @classmethod
    def get_instance(cls, group: Optional[dist.ProcessGroup] = None) -> CacheRoute:
        if cls._instance is None:
            cls._instance = CacheRoute(group)
        return cls._instance