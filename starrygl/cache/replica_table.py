import torch

class CSRReplicaTable:
    def __init__(self, indptr: torch.Tensor, indices: torch.Tensor, locs: torch.Tensor, device: torch.device = torch.device('cpu')):
        """
        Args:
            indptr: [N+1] int64/int32, CSR 行指针
            indices: [E] int64/int32, CSR 列索引 (存储目标 Rank ID)
        """
        self.indptr = indptr
        self.indices = indices
        self.locs = locs
        self.device = indptr.device
        

    def lookup(self, batch_ids: torch.Tensor):
        """
        向量化查询函数，适配 CacheRoute 的接口要求。
        
        Args:
            batch_ids: [B] 当前 batch 的全局节点 ID
            
        Returns:
            src_indices: [Total_Out] 指向 batch_ids (以及 features) 的下标 (0 ~ B-1)
            target_ranks: [Total_Out] 对应的目标 Rank
        """
        # 获取每个 ID 对应的副本区间 [start, end)
        starts = self.indptr[batch_ids]
        ends = self.indptr[batch_ids + 1]
        lengths = ends - starts  # 每个节点有多少个副本
        
        total_replicas = lengths.sum().item()
        
        # 如果没有副本需要发送
        if total_replicas == 0:
            return (torch.empty(0, dtype=torch.long, device=self.device), 
                    torch.empty(0, dtype=torch.long, device=self.device))

        # 2. 生成 src_indices (用于在 CacheRoute 中做 replicate)
        # 作用：告诉 CacheRoute，features 的第 k 行需要被复制 lengths[k] 次
        # 逻辑：repeat_interleave([0, 1], [2, 3]) -> [0, 0, 1, 1, 1]
        batch_range = torch.arange(batch_ids.size(0), device=self.device)
        src_indices = torch.repeat_interleave(batch_range, lengths)

        # 3. 生成 target_ranks (从 CSR indices 中 gather 数据)
        # 这是一个经典的 "Segmented Gather" 技巧，完全避免 Python 循环
        
        # A. 扩展 starts: [s0, s0, s1, s1, s1]
        expanded_starts = torch.repeat_interleave(starts, lengths)
        
        # B. 生成段内偏移 (Inner Offsets): [0, 1, 0, 1, 2]
        # 方法：全局 Range - 扩展后的段起始 Range
        #   Range: [0, 1, 2, 3, 4]
        #   Offset_Base: [0, 2] (lengths 的累加, shift 1位)
        #   Expanded_Base: [0, 0, 2, 2, 2]
        #   Result: [0, 1, 2, 3, 4] - [0, 0, 2, 2, 2] = [0, 1, 0, 1, 2]
        
        # B1. 计算 batch 内的累积长度作为 Base
        # 注意：这里需要在 batch 维度做 cumsum，而不是用全局的 indptr
        cumsum_lengths = torch.cumsum(lengths, dim=0)
        # 构造 [0, len0, len0+len1, ...]
        offset_base = torch.cat([
            torch.tensor([0], device=self.device, dtype=lengths.dtype), 
            cumsum_lengths[:-1]
        ])
        
        # B2. 扩展 Base
        expanded_offset_base = torch.repeat_interleave(offset_base, lengths)
        
        # B3. 生成 0~Total-1 的序列
        flat_range = torch.arange(total_replicas, device=self.device)
        
        # B4. 得到 inner_offsets
        inner_offsets = flat_range - expanded_offset_base
        
        # C. 最终的 CSR 索引
        gather_indices = expanded_starts + inner_offsets
        
        # D. 取值
        target_ranks = self.indices[gather_indices]
        target_locs = self.locs[gather_indices]
        return src_indices, target_ranks, target_locs
    


class UVACSRReplicaTable:

    def __init__(self, indptr: torch.Tensor, indices: torch.Tensor, max_batch_size_estimate=100000):
        """
        Args:
            indptr: [N+1] int64/int32, CSR 行指针
            indices: [E] int64/int32, CSR 列索引 (存储目标 Rank ID)
        """
        self.indptr = indptr.cuda()
        if indices.is_cuda:
            indices = indices.cpu()
            locs = locs.cpu()
        self.indices = indices.contiguous().pin_memory()
        self.locs = locs.contiguous().pin_memory()
        self.staging_buffer = torch.empty(
            max_batch_size_estimate, 
            dtype=indices.dtype
        ).pin_memory()

    def lookup(self, batch_ids: torch.Tensor):
        """
        混合查询：GPU 算地址 -> CPU 取数据 -> GPU 拿结果
        """
        # =========================================================
        # 阶段 A: GPU 高速计算 (完全在 VRAM 内)
        # =========================================================
        # 1. 计算区间
        starts = self.indptr[batch_ids]
        ends = self.indptr[batch_ids + 1]
        lengths = ends - starts
        
        total_replicas = lengths.sum().item()
        if total_replicas == 0:
            return (torch.empty(0, dtype=torch.long, device=self.device), 
                    torch.empty(0, dtype=torch.long, device=self.device))

        # 2. 生成 src_indices (用于 features 复制)
        # 这一步必须在 GPU 做，因为 features 在 GPU
        batch_range = torch.arange(batch_ids.size(0), device=self.device)
        src_indices = torch.repeat_interleave(batch_range, lengths)

        # 3. 计算 CSR 偏移量 (Segmented Gather 算法)
        # 这步数学运算量大，在 GPU 做比 CPU 快得多
        expanded_starts = torch.repeat_interleave(starts, lengths)
        
        # 计算 inner offsets
        cumsum_lengths = torch.cumsum(lengths, dim=0)
        offset_base = torch.cat([
            torch.tensor([0], device=self.device, dtype=lengths.dtype), 
            cumsum_lengths[:-1]
        ])
        expanded_offset_base = torch.repeat_interleave(offset_base, lengths)
        
        flat_range = torch.arange(total_replicas, device=self.device)
        inner_offsets = flat_range - expanded_offset_base
        
        # 得到最终指向 self.indices 的索引 (GPU Tensor)
        gather_indices_gpu = expanded_starts + inner_offsets

        # =========================================================
        # 阶段 B: UVA/Pinned Memory 读取
        # =========================================================
        
        # 1. 将索引传回 CPU (Async)
        # 由于 indices 在 CPU 上，我们需要用 CPU index 去取
        gather_indices_cpu = gather_indices_gpu.to('cpu', non_blocking=True)
        required_size = gather_indices_cpu.size(0) * 2
        if required_size > self.staging_buffer.size(0):
            self.staging_buffer = torch.empty(
                int(required_size * 1.5), # 扩容 1.5 倍
                dtype=self.indices.dtype
            ).pin_memory()
        current_buffer = self.staging_buffer[:required_size]
        
        torch.index_select(
            self.indices,      
            0,                 
            gather_indices_cpu,
            out=current_buffer[:gather_indices_cpu.size(0)]
        )
        torch.index_select(
            self.locs,
            0,
            gather_indices_cpu,
            out=current_buffer[gather_indices_cpu.size(0):]
        )
        target_ranks = current_buffer[:gather_indices_cpu.size(0)].to(self.device, non_blocking=True)
        target_locs = current_buffer[gather_indices_cpu.size(0):].to(self.device, non_blocking=True)

        return src_indices, target_ranks, target_locs
    
    
def build_replica_table(num_nodes, partition_book, num_parts, type = None):
    indptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    indices = []
    halo_list = []
    for part in range(num_parts):
        nodes = partition_book[part]
        # 确保是 Tensor
        if not isinstance(nodes, torch.Tensor):
            nodes = torch.tensor(nodes, dtype=torch.long)
        else:
            assert nodes.dtype == torch.long, "Nodes must be long tensor"
        part_ids = torch.full_like(nodes, part, dtype=torch.long)
        loc = torch.arange(nodes.shape[0], dtype=torch.long, device=nodes.device)
        halo_list.append(torch.stack((nodes, part_ids, loc)))
    if not halo_list:
        # 处理空列表边界情况
        all_pairs = torch.empty((3, 0), dtype=torch.long)
    else:
        all_pairs = torch.cat(halo_list, dim=1)
    sorted_unique_pairs = all_pairs.unique(dim=1)
    indices = sorted_unique_pairs[1, :]
    locs = sorted_unique_pairs[2, :]
    counts = sorted_unique_pairs[0, :].bincount(minlength=num_nodes)
    indptr[1:] = torch.cumsum(counts, dim=0)
    if type == 'UVA':
        # UVACSRReplicaTable 需要 indices 在 CPU 上
        return UVACSRReplicaTable(
            indptr=indptr,
            indices=indices.cpu(),
            locs=locs.cpu(),
            max_batch_size_estimate=0.2 * num_nodes
        )
    return CSRReplicaTable(
        indptr=indptr, 
        indices=indices,
        locs=locs
    )
    

