import torch
import dgl
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Iterator
from torch.utils.data import Dataset, Sampler

# 引入通信计划定义
from starrygl.cache.cache_route import CommPlan

# ==============================================================================
# 1. 数据容器 (Data Containers)
# ==============================================================================

@dataclass
class AtomicBatch:
    """
    原子批次：对应磁盘上的一个 .pt 文件 (即一个 Chunk)。
    包含了该 Chunk 的任务数据、每一层的子图结构以及通信路由。
    """
    task_data: Dict[str, torch.Tensor]          # 如 task_src, task_dst, task_ts, labels
    layer_data: List[Dict[str, torch.Tensor]]   # 每一层的 CSC/CSR 数据
    comm_plans: List[Optional[CommPlan]]        # 每一层的通信计划
    num_set = 0
    
    @classmethod
    def load(cls, path: Union[str, Path], num_set:int):
        """从磁盘加载 .pt 文件并反序列化"""
        if isinstance(path,str):
            raw_data = torch.load(path, map_location='cpu') 
            if isinstance(raw_data, list):
                task_data = raw_data[0]
                layer_data = raw_data[1:]
            plans = task_data.get('comm_plans', task_data.get('comm_plan', task_data.get('route', [])))
            if isinstance(plans, CommPlan) or plans is None:
                plans = [plans]
            return cls(task_data, layer_data, plans)
        elif isinstance(path,tuple):
            raw_data = torch.load(path, map_location='cpu') 
            neg_data = torch.load(path, map_location='cpu') 
            if isinstance(raw_data, list):
                task_data = raw_data[0]
                layer_data = raw_data[1:]
                neg_task_data = neg_data[0]
                neg_layer_data = neg_data[1:]
            plans = task_data.get('comm_plans', task_data.get('comm_plan', task_data.get('route', [])))
            neg_plans = neg_task_data.get('comm_plans', neg_task_data.get('comm_plan', neg_task_data.get('route', [])))
            if isinstance(plans, CommPlan) or plans is None:
                plans = [plans]
            elif isinstance(plans, list):
                pass # 已经是 list
            if isinstance(neg_plans,CommPlan) or neg_plans is None:
                neg_plans = [neg_plans]
            return cls([task_data, neg_task_data], [layer_data, neg_layer_data], [plans, neg_plans])
        else:
            raise ValueError(f"Unknown data format in {path}")
        


class MergedBatch:
    """
    合并后的批次：由多个 AtomicBatch 拼接而成。
    这个对象通常在 collate_fn 中创建，并负责将数据移动到 GPU 以及构建 DGL Blocks。
    """
    def __init__(self, task_data, layer_data, comm_plans):
        self.task_data = task_data
        self.layer_data = layer_data
        self.comm_plans = comm_plans # List[CommPlan] 对应每一层
        self._blocks = None          # 缓存构建好的 DGLBlock

    def to(self, device, non_blocking=True):
        """
        将 batch 内的所有 Tensor 移动到指定设备 (支持异步)。
        """
        # 1. Task Data
        for k, v in self.task_data.items():
            if isinstance(v, torch.Tensor): 
                self.task_data[k] = v.to(device, non_blocking=non_blocking)
        
        # 2. Layer Data
        for l in self.layer_data:
            for k, v in l.items():
                if isinstance(v, torch.Tensor): 
                    l[k] = v.to(device, non_blocking=non_blocking)
                
        # 3. Route Plans
        for plan in self.comm_plans:
            if plan is not None:
                plan.to(device, non_blocking=non_blocking)
                
        return self

    @property
    def blocks(self):
        """
        Lazy construction of DGL Blocks on the current device.
        构建顺序：Message Passing 的流向 (Input Layer -> Output Layer)。
        """
        if self._blocks is not None: return self._blocks
        
        blks = []
        for i in reversed(range(len(self.layer_data))):
            data = self.layer_data[i]
            
            # CSC 格式: (indptr, indices, edge_ids)
            # indptr 长度 = num_dst + 1
            num_dst = len(data['indptr']) - 1
            num_src = len(data['gids']) # gids 是这一层的源节点 Global ID
            
            # 使用 create_block 高效构建
            g = dgl.create_block(
                ('csc', (data['indptr'], data['indices'], torch.tensor([]))),
                num_src_nodes=num_src,
                num_dst_nodes=num_dst,
                device=data['indptr'].device # 数据在哪，图就在哪
            )
            
            # 填充必要的元数据
            g.srcdata[dgl.NID] = data['gids']
            # g.dstdata[dgl.NID] 不需要显式设置，它隐式等于上一层的 srcdata[dgl.NID]
            
            if 'edge_ts' in data: g.edata['ts'] = data['edge_ts']
            if 'eid' in data: g.edata[dgl.EID] = data['eid']
            
            blks.append(g)
            
        self._blocks = blks
        return blks

# ==============================================================================
# 2. Dataset & Sampler
# ==============================================================================

class AtomicDataset(Dataset):
    """
    轻量级 Dataset，只负责索引磁盘上的 Chunk 文件路径。
    不进行实际的数据加载，加载由 DataLoader 的 Worker 在 collate_fn 中完成。
    """
    def __init__(self, root_dir: Union[str, Path], neg_set:int):
        self.root = Path(root_dir)
        self.neg_set = 0
        if not self.root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.root}")
            
        self.files = sorted(list(self.root.glob("slot_*.pt")), key=lambda x: x.name)
        
        self.slot_groups = {}
        for idx, f in enumerate(self.files):
            try:
                parts = f.name.split('_')
                slot_id = int(parts[1])
                
                if slot_id not in self.slot_groups:
                    self.slot_groups[slot_id] = []
                self.slot_groups[slot_id].append(idx)
            except (IndexError, ValueError): 
                continue
                
        self.sorted_slots = sorted(self.slot_groups.keys())
        print(f"Dataset loaded: {len(self.files)} chunks across {len(self.sorted_slots)} slots.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 只返回路径字符串
        return str(self.files[idx])

class SlidingWindowDataset(Dataset):
    def __init__(self, root_dir: str, window_size: int = 2, step: int = 1):
        self.root = Path(root_dir)
        self.files = sorted(list(self.root.glob("slot_*.pt")), key=lambda x: x.name)
        self.window_size = window_size
        self.step = step
        
        # 构建窗口
        self.windows = []
        num_files = len(self.files)
        if num_files >= window_size:
            for i in range(0, num_files - window_size + 1, step):
                # 每个 Item 是一个文件路径列表 [path_t0, path_t1, ...]
                self.windows.append([str(f) for f in self.files[i : i + window_size]])
        
        print(f"SlidingWindowDataset: {len(self.windows)} windows (size={window_size})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # 返回 List[str]
        return self.windows[idx]
    
class SlotAwareSampler(Sampler):
    """
    [时序感知采样器]
    1. Inter-Slot Order: 严格按照时间槽 (Slot 0 -> Slot 1 -> ...) 顺序遍历。
    2. Intra-Slot Shuffle: 在同一个时间槽内部，随机打乱 Sub-chunks 的顺序。
    
    这保证了训练的时序因果性，同时增加了 SGD 所需的随机性。
    """
    def __init__(self, dataset: AtomicDataset):
        self.dataset = dataset

    def __iter__(self) -> Iterator[int]:
        for slot_id in self.dataset.sorted_slots:
            indices = self.dataset.slot_groups[slot_id]
            # Slot 内部 Shuffle
            # 使用 torch.randperm 生成随机索引
            perm = torch.randperm(len(indices)).tolist()
            
            for i in perm:
                if self.dataset.neg_set > 0:
                    file = indices[i]
                    neg_ids = torch.randint(0, self.dataset.neg_set).item()
                    neg_file = '/'.join(file.split('/')[:-1] + [f'neg_{neg_ids}_'+file.split('/')[-1]])
                    yield (file, neg_file)
                yield indices[i]

    def __len__(self):
        return len(self.dataset)


def collate_and_merge(file_paths: List[Union[str, tuple]]) -> Union[MergedBatch, None]:
    """
    DataLoader 的 collate_fn (增强版)。
    
    1. Flatten: 拍平正负样本。
    2. Task Merge: 特别处理 gids 和 inv_map 的偏移合并。
    3. Layer Merge: CSC 拼接。
    4. Route Merge: 通信计划 Re-indexing。
    """
    if len(file_paths) == 0: return None
    
    # --- 1. Load & Flatten Batches ---
    raw_batches = [AtomicBatch.load(f, num_set=0) for f in file_paths]
    if not raw_batches: return None
    # [Flatten] 拍平正负样本 List
    flat_batches = []
    neg_flat_batches = []
    for b in raw_batches:
        if isinstance(b.task_data, list):
            for i in range(len(b.task_data)):
                plans = b.comm_plans[i] if isinstance(b.comm_plans, list) else [b.comm_plans]
                if(b.task_data[i]['task_label'] == 'neg_set'):
                    b.task_data[i]['task_neg_dst'] = b.task_data[i].pop('task_nodes')
                    b.task_data[i]['task_ts'] = b.task_data[i]['task_ts']
                    neg_flat_batches.append(AtomicBatch(b.task_data[i], b.layer_data[i], plans))
                else:
                    flat_batches.append(AtomicBatch(b.task_data[i], b.layer_data[i], plans))
        else:
            flat_batches.append(b)

    if not flat_batches: return None
    flat_batches.extend(neg_flat_batches) 
    # --- 2. Merge Task Data (支持 gids + inv_map 偏移) ---
    merged_task = {}
    example_task = flat_batches[0].task_data
    
    # 特殊处理的 Key
    special_keys = ['gids', 'inv_map']
    ignored_keys = ['comm_plans', 'comm_plan', 'route']
    
    # 检查是否存在需要偏移合并的 gids/inv_map
    has_gids = 'gids' in example_task
    has_inv_map = 'inv_map' in example_task
    gids_len = []
    current_gid_offset = 0
    current_total_gid_offset = 0
    if has_gids:
        gids_list = []
        total_gids_list = []
        inv_map_list = []
        mem_indices_list = []
        for b in flat_batches:
            # GIDs: 收集
            batch_gids = b.task_data['gids']
            gids_list.append(batch_gids)
            gids_len.append(batch_gids.shape[0])
            if has_inv_map:
                batch_inv = b.task_data['inv_map']
                inv_map_list.append(batch_inv + current_gid_offset)
                mem_indices_list.append(batch_inv + current_total_gid_offset)
            current_gid_offset += batch_gids.shape[0]
            current_total_gid_offset += b.layer_data[-1]['gids'].shape[0]
        merged_task['gids'] = torch.cat(gids_list)
        if has_inv_map:
            merged_task['inv_map'] = torch.cat(inv_map_list)
            merged_task['mem_inv_map'] = torch.cat(mem_indices_list)

    # 处理其他常规 Tensor (直接拼接)
    for k in example_task.keys():
        if k not in ignored_keys and k not in special_keys:
            vals = [b.task_data[k] for b in flat_batches]
            if isinstance(vals[0], torch.Tensor):
                merged_task[k] = torch.cat(vals, dim=0)
            else:
                merged_task[k] = vals


    # --- 3. Merge Layer Data (CSC Stitching) ---
    merged_layers = []
    new_ind_offset = [torch.arange(data.layer_data[-1]['gids'].shape[0]) for data in flat_batches] 

    if flat_batches[0].layer_data:
        num_layers = len(flat_batches[0].layer_data)
        batch_input_node_counts = [] # 用于 Route Re-indexing
        for l in range(num_layers):
            l_data = {}
            indptr_list = []
            indices_list = []
            eid_list = []
            ts_list = []
            src_indices_list = []
            #gids_list = [merged_task['gids']] if has_gids else merged_layers[-1]['gids']
            gids_list = []
            
            current_offset = 0
            layer_node_counts = [] # 这一层每个batch的节点数
            current_gid_offset = 0
            for b_idx,b in enumerate(flat_batches):
                layer = b.layer_data[l]
                layer_node_counts.append(layer['gids'].shape[0])

                # Indptr 偏移修正
                ptr = layer['indptr']
                if len(indptr_list) == 0:
                    indptr_list.append(ptr)
                else:
                    indptr_list.append(ptr[1:] + current_offset)
                gids_list.append(layer['gids'])
                current_offset += ptr[-1].item()    
                src_indices_list.append(torch.arange(len(layer['indices'])) + current_gid_offset)
                indices_list.append(layer['indices'] +  current_gid_offset)
                current_gid_offset += layer['gids'].shape[0]
                if 'eid' in layer: eid_list.append(layer['eid'])
                if 'edge_ts' in layer: ts_list.append(layer['edge_ts'])
            l_data['indptr'] = torch.cat(indptr_list)
            l_data['indices'] = torch.cat(indices_list)
            l_data['gids'] = torch.cat(gids_list)
            l_data['src_indices'] = torch.cat(src_indices_list)
            if eid_list: l_data['eid'] = torch.cat(eid_list)
            if ts_list: l_data['edge_ts'] = torch.cat(ts_list)
            
            merged_layers.append(l_data)
            batch_input_node_counts.append(layer_node_counts)

    # --- 4. Merge Route Plans (Re-indexing) ---
    merged_plans = []
    example_plans = flat_batches[0].comm_plans
    if example_plans is None or isinstance(example_plans, CommPlan):
        for l in example_plans:
            send_sizes = torch.zeros_like(example_plans[0].send_sizes)
            send_locs = [[] for i in range(torch.distributed.get_world_size())]
            remote_locs = [[] for i in range(torch.distributed.get_world_size())]
            send_ranks = [[] for i in range(torch.distributed.get_world_size())]
            for b_idx, b in enumerate(flat_batches):
                if l is None or l >= len(b.comm_plans): continue
                plan:CommPlan = b.comm_plans[l]
                if plan is None: continue
                # Re-indexing
                #new_send_locs = new_ind_offset[plan.send_indices]
                curr_rank_start = 0
                send_sizes += plan.send_sizes
                for r in range(torch.distributed.get_world_size()):
                    curr_rank_end = curr_rank_start + plan.send_sizes + plan.send_sizes[r]
                    send_locs[r].append(plan.send_indices[curr_rank_start:curr_rank_end])
                    #send_locs[r].extend(new_send_locs[new_send_locs[curr_rank_start:curr_rank_end]])
                    remote_locs[r].append(plan.send_remote_indices[curr_rank_start:curr_rank_end])
                    send_ranks[r].append(plan.send_ranks[curr_rank_start:curr_rank_end])
                    curr_rank_start = curr_rank_end
            new_ranks = [torch.cat(send_ranks[r]) for r in range(torch.distributed.get_world_size())]
            new_indices = [torch.cat(send_locs[r]) for r in range(torch.distributed.get_world_size())]
            new_remote_indices = [torch.cat(remote_locs[r]) for r in range(torch.distributed.get_world_size())]
            new_ranks = torch.cat(new_ranks)
            new_indices = torch.cat(new_indices)
            new_remote_indices = torch.cat(new_remote_indices)
            merged_plans.append(CommPlan(new_ranks, send_sizes, new_indices, new_remote_indices))

    return MergedBatch(merged_task, merged_layers, merged_plans)



def collate_sliding_window(windows: List[List[str]]) -> Union[MergedBatch, None]:
    """
    处理滑动窗口的合并逻辑。
    假设 DataLoader 的 batch_size=1 (即一次处理一个窗口 [T0, T1...])。
    如果 batch_size > 1，需要先 flatten windows。
    """
    if len(windows) == 0: return None
    
    # DataLoader batch_size=1 时，windows 是 [['f1', 'f2']]，需要解包
    # 如果 batch_size > 1，windows 是 [['f1', 'f2'], ['f2', 'f3']]，也flatten处理
    file_paths = [f for w in windows for f in w] 
    
    # --- 1. Load & Flatten (复用之前的逻辑) ---
    raw_batches = [AtomicBatch.load(f, num_set=0) for f in file_paths]
    flat_batches = []
    
    for b in raw_batches:
        if isinstance(b.task_data, list):
            # Pos
            pos_plans = b.comm_plans[0] if isinstance(b.comm_plans, list) else b.comm_plans
            flat_batches.append(AtomicBatch(b.task_data[0], b.layer_data[0], pos_plans))
            # Neg
            neg_data = b.task_data[1]
            if neg_data.get('task_label') == 'neg_set':
                # Hydrate Neg Data (简略版，假设已补全 src/ts)
                neg_data['task_dst'] = neg_data.pop('task_nodes') 
                # ... (补全逻辑同前) ...
                neg_plans = b.comm_plans[1] if isinstance(b.comm_plans, list) else None
                flat_batches.append(AtomicBatch(neg_data, b.layer_data[1], neg_plans))
        else:
            flat_batches.append(b)
            
    if not flat_batches: return None

    # =========================================================
    # Phase 2: Global Deduplication (构建 Super GIDs)
    # =========================================================
    
    # 1. 收集所有 Batch 的 GIDs
    all_gids_list = []
    batch_gid_ranges = [] # 记录每个 batch 在 cat 后的区间 [start, end)
    curr_start = 0
    
    # 为了处理 Task 层和 Layer 层可能都需要映射，我们通常对所有涉及的 GID 去重
    # 这里假设: 所有层用的 GID 都在 task_data['gids'] (Layer 0 Input) 里或者 layer_data[-1] 里
    # 在 TGN/StarryGL 中，通常每一层都有自己的 gids。
    # 为了简单，我们只对 "特征查找层" (通常是 Layer 0 的 gids 或 Layer N 的 gids) 做去重。
    # 假设：我们需要对 Task 数据中涉及的所有 gids 做统一去重。
    
    for b in flat_batches:
        g = b.task_data['gids']
        all_gids_list.append(g)
        batch_gid_ranges.append((curr_start, curr_start + g.shape[0]))
        curr_start += g.shape[0]
        
    all_gids_cat = torch.cat(all_gids_list)
    
    # 2. 计算 Unique (Super GIDs) 和 Inverse Map
    # return_inverse=True: 返回原数据在 unique 数组中的下标
    super_gids, inverse_indices = torch.unique(all_gids_cat, return_inverse=True, sorted=True)
    
    # 3. 切分 Map 给每个 Batch
    # batch_maps[i] 是一个 Tensor: Batch i 的第 k 个本地节点 -> Super GID 的全局下标
    batch_maps = []
    for start, end in batch_gid_ranges:
        batch_maps.append(inverse_indices[start:end])

    # =========================================================
    # Phase 3: Merge Task Data (应用映射)
    # =========================================================
    merged_task = {}
    
    # GIDs 直接替换为去重后的 Super GIDs
    merged_task['gids'] = super_gids
    
    # Inv Map 需要重映射
    # 原本: inv_map 指向 Local GIDs
    # 现在: inv_map 必须指向 Super GIDs
    if 'inv_map' in flat_batches[0].task_data:
        new_inv_maps = []
        for b_idx, b in enumerate(flat_batches):
            old_inv = b.task_data['inv_map']
            local_map = batch_maps[b_idx]
            # [关键]: 查表转换 ID
            new_inv = local_map[old_inv]
            new_inv_maps.append(new_inv)
        merged_task['inv_map'] = torch.cat(new_inv_maps)
        
    # 其他 Task 字段 (src, dst, ts, label) 直接 concat，不受影响
    # ... (常规 concat 代码略) ...

    # =========================================================
    # Phase 4: Merge Layer Data (应用映射)
    # =========================================================
    merged_layers = []
    if flat_batches[0].layer_data:
        num_layers = len(flat_batches[0].layer_data)
        
        for l in range(num_layers):
            # 注意：如果 GNN 是多层的，每一层的 GIDs 集合可能不同。
            # 如果每一层都做 Unique，计算量很大。
            # 简化策略：通常只对 Input Layer (Layer 0) 做 Unique (特征查找层)。
            # 中间层的 GIDs 可以不做 Unique，只是简单的 Block Diagonal 堆叠。
            # 下面代码演示：Layer Indices 指向 Super GIDs (假设 Indices 指向的是 Layer 0 GIDs)
            
            l_data = {}
            indptr_list = []
            indices_list = []
            gids_list = [] # 这里的 gids 存什么？
            
            # 如果是 Block Diagonal 堆叠，Indptr 还是简单的累加
            curr_indptr_off = 0
            
            for b_idx, b in enumerate(flat_batches):
                layer = b.layer_data[l]
                
                # 1. Indptr 常规合并
                ptr = layer['indptr']
                if not indptr_list: indptr_list.append(ptr)
                else: indptr_list.append(ptr[1:] + curr_indptr_off)
                curr_indptr_off += ptr[-1].item()
                
                # 2. Indices 重映射 (核心)
                # CSC Indices 指向的是 Source Nodes (GIDs)
                # 我们希望所有 Indices 都指向 Super GIDs
                raw_idx = layer['indices']
                local_map = batch_maps[b_idx]
                
                # [关键]: 将指向 Local GID 的边，改为指向 Super GID
                new_idx = local_map[raw_idx]
                indices_list.append(new_idx)
                
                # 3. GIDs
                # 因为 Indices 已经指向了全局 Super GIDs，
                # 这一层的 gids 属性其实变成了 super_gids (对于 Layer 0)
                # 或者是局部 gids (对于中间层，如果中间层不做去重)
                # 这里我们暂且不存 gids_list，最后统一用 super_gids
                pass

            l_data['indptr'] = torch.cat(indptr_list)
            l_data['indices'] = torch.cat(indices_list)
            l_data['gids'] = super_gids # 只有一份全局 GID
            
            # (eid, ts concat 略)
            merged_layers.append(l_data)

    # =========================================================
    # Phase 5: Merge Route Plans (应用映射)
    # =========================================================
    merged_plans = []
    
    # 逻辑同前，但这次使用 batch_maps 来修正 send_indices
    # ... 获取 example_plans ...
    
    if example_plans:
        for i in range(len(example_plans)):
            # ... valid_plans loop ...
            
            m_idx = []
            # ...
            for k, plan in enumerate(valid_plans):
                b_idx = valid_batch_indices[k]
                
                # [关键]: 使用 Map 修正本地索引
                # 使得 Route 直接去 Super GIDs 的位置取特征
                local_map = batch_maps[b_idx]
                new_idx = local_map[plan.send_indices]
                m_idx.append(new_idx)
                
                # ... (ranks, remote 拼接) ...
            
            # ... (New Plan) ...

    return MergedBatch(merged_task, merged_layers, merged_plans)



# 假设已导入 AtomicBatch, MergedBatch, CommPlan

def collate_sliding_window(windows: List[List[str]]) -> Union[MergedBatch, None]:
    """
    DataLoader Collate Function for Sliding Window (e.g. TGN/TGAT).
    Merges multiple time-slots into a single Super-Batch with globally unique nodes.
    """
    if len(windows) == 0: return None
    
    # 0. Flatten Windows (如果 batch_size > 1)
    # 假设 Input 是 [['f1', 'f2'], ['f3', 'f4']] -> ['f1', 'f2', 'f3', 'f4']
    file_paths = [f for w in windows for f in w]
    if not file_paths: return None

    # =========================================================
    # Phase 1: Load & Flatten & Hydrate (基础数据加载)
    # =========================================================
    raw_batches = [AtomicBatch.load(f, num_set=0) for f in file_paths]
    flat_batches = []
    
    for b in raw_batches:
        if isinstance(b.task_data, list):
            # Pos Batch
            pos_plans = b.comm_plans[0] if isinstance(b.comm_plans, list) else b.comm_plans
            flat_batches.append(AtomicBatch(b.task_data[0], b.layer_data[0], pos_plans))
            
            # Neg Batch
            neg_data = b.task_data[1]
            if neg_data.get('task_label') == 'neg_set':
                # Hydrate: 补全 src, ts 等字段以匹配 Pos 结构
                pos_data = b.task_data[0]
                num_pos = pos_data['task_src'].shape[0]
                num_neg = neg_data['task_nodes'].shape[0]
                ratio = num_neg // num_pos
                
                neg_data['task_src'] = pos_data['task_src'].repeat_interleave(ratio)
                neg_data['task_dst'] = neg_data.pop('task_nodes') # Rename to dst
                
                # 补全 TS (如果缺失)
                if 'task_ts' not in neg_data or neg_data['task_ts'].shape[0] != num_neg:
                    neg_data['task_ts'] = pos_data['task_ts'].repeat_interleave(ratio)
                
                neg_plans = b.comm_plans[1] if isinstance(b.comm_plans, list) else None
                flat_batches.append(AtomicBatch(neg_data, b.layer_data[1], neg_plans))
        else:
            flat_batches.append(b)
            
    if not flat_batches: return None

    # =========================================================
    # Phase 2: Global Deduplication (构建 Super GIDs)
    # =========================================================
    all_gids_list = []
    batch_gid_ranges = [] # 记录每个 batch 在 cat 后的区间 [start, end)
    curr_start = 0
    
    # 收集 Task 层涉及的所有 GIDs (通常 Layer 0 的 Input GIDs 包含在 task_data['gids'] 中)
    for b in flat_batches:
        g = b.task_data['gids']
        all_gids_list.append(g)
        batch_gid_ranges.append((curr_start, curr_start + g.shape[0]))
        curr_start += g.shape[0]
        
    all_gids_cat = torch.cat(all_gids_list)
    
    # [核心] 计算 Unique (Super GIDs) 和 Inverse Indices
    # inverse_indices: 原数组中的元素对应 super_gids 中的哪个下标
    super_gids, inverse_indices = torch.unique(all_gids_cat, return_inverse=True, sorted=True)
    
    # 将长长的 inverse_indices 切分回每个 Batch，形成 Local->Global 映射表
    batch_maps = []
    for start, end in batch_gid_ranges:
        batch_maps.append(inverse_indices[start:end])

    # =========================================================
    # Phase 3: Merge Task Data (应用映射)
    # =========================================================
    merged_task = {}
    example_task = flat_batches[0].task_data
    
    # 1. GIDs: 替换为 Super GIDs
    merged_task['gids'] = super_gids
    
    # 2. Inv Map: 重映射 (Local ID -> Super ID)
    if 'inv_map' in example_task:
        new_inv_maps = []
        for b_idx, b in enumerate(flat_batches):
            old_inv = b.task_data['inv_map']
            local_map = batch_maps[b_idx]
            # 查表转换
            new_inv_maps.append(local_map[old_inv])
        merged_task['inv_map'] = torch.cat(new_inv_maps)
    
    # 3. 其他常规字段 (直接 Concat)
    special_keys = ['gids', 'inv_map', 'comm_plans', 'comm_plan', 'route']
    for k in example_task.keys():
        if k not in special_keys:
            try:
                vals = [b.task_data[k] for b in flat_batches]
                if isinstance(vals[0], torch.Tensor):
                    merged_task[k] = torch.cat(vals, dim=0)
                else:
                    merged_task[k] = vals
            except: pass

    # =========================================================
    # Phase 4: Merge Layer Data (应用映射)
    # =========================================================
    merged_layers = []
    
    if flat_batches[0].layer_data:
        num_layers = len(flat_batches[0].layer_data)
        
        for l in range(num_layers):
            l_data = {}
            indptr_list = []
            indices_list = []
            eid_list = []
            ts_list = []
            
            # 这里采用简单的 Block Diagonal 堆叠 Indptr
            curr_indptr_off = 0
            
            for b_idx, b in enumerate(flat_batches):
                layer = b.layer_data[l]
                
                # 1. Indptr (Row Ptr) 常规合并
                ptr = layer['indptr']
                if not indptr_list: indptr_list.append(ptr)
                else: indptr_list.append(ptr[1:] + curr_indptr_off)
                curr_indptr_off += ptr[-1].item()
                
                # 2. Indices (Col Idx) 重映射 [核心]
                # CSC Indices 原本指向 Local GIDs
                # 现在必须指向 Super GIDs
                raw_idx = layer['indices']
                local_map = batch_maps[b_idx]
                
                # 查表重连: Local -> Super
                new_idx = local_map[raw_idx]
                indices_list.append(new_idx)
                
                # 收集边属性
                if 'eid' in layer: eid_list.append(layer['eid'])
                if 'edge_ts' in layer: ts_list.append(layer['edge_ts'])

            l_data['indptr'] = torch.cat(indptr_list)
            l_data['indices'] = torch.cat(indices_list)
            # 这一层的节点集合就是 Super GIDs (所有层共享同一套节点池)
            l_data['gids'] = super_gids 
            
            if eid_list: l_data['eid'] = torch.cat(eid_list)
            if ts_list: l_data['edge_ts'] = torch.cat(ts_list)
            
            merged_layers.append(l_data)

    # =========================================================
    # Phase 5: Merge Route Plans (应用映射)
    # =========================================================
    merged_plans = []
    
    # 寻找非空 Plan 模板
    example_plans = None
    for b in flat_batches:
        if b.comm_plans:
            example_plans = b.comm_plans if isinstance(b.comm_plans, list) else [b.comm_plans]
            break
            
    if example_plans:
        num_plans = len(example_plans)
        
        for i in range(num_plans):
            valid_plans = []
            valid_batch_indices = []
            
            for b_idx, b in enumerate(flat_batches):
                plans = b.comm_plans if isinstance(b.comm_plans, list) else [b.comm_plans]
                if i < len(plans) and plans[i] is not None:
                    valid_plans.append(plans[i])
                    valid_batch_indices.append(b_idx)
            
            if not valid_plans:
                merged_plans.append(None)
                continue
            
            m_ranks, m_idx, m_rem = [], [], []
            m_sizes_list = []
            
            for k, plan in enumerate(valid_plans):
                b_idx = valid_batch_indices[k]
                local_map = batch_maps[b_idx] # 获取该 Batch 的映射表
                
                m_ranks.append(plan.send_ranks)
                m_sizes_list.append(plan.send_sizes)
                m_rem.append(plan.send_remote_indices) # 远端物理地址不变
                
                # [关键]: 修正本地索引 -> 指向 Super GIDs
                # Route 拿回来的特征，直接填入 Super GIDs 对应的位置
                # 注意: send_indices 必须在 CPU 上才能查表 (如果 map 在 CPU)
                new_send_idx = local_map[plan.send_indices]
                m_idx.append(new_send_idx)
                
            new_plan = CommPlan(
                torch.cat(m_ranks),
                torch.stack(m_sizes_list).sum(dim=0),
                torch.cat(m_idx),
                torch.cat(m_rem)
            )
            merged_plans.append(new_plan)
    else:
        merged_plans = []

    return MergedBatch(merged_task, merged_layers, merged_plans)