import torch
import dgl
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Iterator
from torch.utils.data import Dataset, Sampler

# 引入通信计划定义
from starrygl.cache.cache_route import CommPlan
from starrygl.utils.context import DistributedContext

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
    def load(cls, path: Union[str, Path]):
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
            raw_data = torch.load(path[0], map_location='cpu') 
            neg_data = torch.load(path[1], map_location='cpu') 
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
            g.src_indices= data['src_indices']
            g.dstdata[dgl.NID] = self.layer_data[i-1]['gids'] if i > 0 else self.task_data['gids']
            # g.dstdata[dgl.NID] 不需要显式设置，它隐式等于上一层的 srcdata[dgl.NID]
            
            if 'ts' in data: 
                g.srcdata['ts'] = data['ts']
                g.dstdata['ts'] = self.layer_data[i-1]['ts'] if i > 0 else self.task_data['ts']
            if 'eid' in data: g.edata[dgl.EID] = data['eid']
            if 'edge_dt' in data: g.edata['dt'] = data['edge_dt']
            
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
        self.neg_set = neg_set
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
        if isinstance(idx,tuple):
            neg_id = idx[1]
            file_path = self.files[idx[0]]
            neg_file = '/'.join(str(file_path).split('/')[:-1] + [f'neg_{neg_id}_'+str(file_path).split('/')[-1]])
            return (str(file_path), str(neg_file))
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
                    neg_ids = torch.randint(low=0, high=self.dataset.neg_set, size=[1]).item()
                    #neg_file = '/'.join(file.split('/')[:-1] + [f'neg_{neg_ids}_'+file.split('/')[-1]])
                    yield (file, neg_ids)
                else:
                    yield indices[i]

    def __len__(self):
        return len(self.dataset)


def collate_and_merge(file_paths: List[Union[str, tuple]], world_size:int) -> Union[MergedBatch, None]:
    """
    DataLoader 的 collate_fn (增强版)。
    
    1. Flatten: 拍平正负样本。
    2. Task Merge: 特别处理 gids 和 inv_map 的偏移合并。
    3. Layer Merge: CSC 拼接。
    4. Route Merge: 通信计划 Re-indexing。
    """
    if len(file_paths) == 0: return None
    
    # --- 1. Load & Flatten Batches ---
    raw_batches = [AtomicBatch.load(f) for f in file_paths]
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
        src_inv_map_list = []
        dst_inv_map_list = []
        src_mem_indices_list = []
        dst_mem_indices_list = []
        inv_map_list = []
        mem_indices_list = []
        for b in flat_batches:
            # GIDs: 收集
            batch_gids = b.task_data['gids']
            gids_list.append(batch_gids)
            gids_len.append(batch_gids.shape[0])
            if has_inv_map:
                batch_inv = b.task_data['inv_map']
                if 'task_src' in b.task_data:
                    num_src = b.task_data['task_src'].shape[0]
                    src_inv_map_list.append(batch_inv[:num_src] + current_gid_offset)
                    src_mem_indices_list.append(batch_inv[:num_src] + current_total_gid_offset)
                    dst_inv_map_list.append(batch_inv[num_src:] + current_gid_offset)
                    dst_mem_indices_list.append(batch_inv[num_src:] + current_total_gid_offset)
                else:
                    inv_map_list.append(batch_inv + current_gid_offset)
                    mem_indices_list.append(batch_inv + current_total_gid_offset)
            current_gid_offset += batch_gids.shape[0]
            current_total_gid_offset += b.layer_data[-1]['gids'].shape[0]
        merged_task['gids'] = torch.cat(gids_list)
        if has_inv_map:
            merged_task['inv_map'] = torch.cat(src_inv_map_list + dst_inv_map_list + inv_map_list)
            merged_task['mem_inv_map'] = torch.cat(src_mem_indices_list + dst_mem_indices_list + mem_indices_list)

   
    # 处理其他常规 Tensor (直接拼接)
    keys = list(example_task.keys()) + ['task_neg_dst']
    for k in keys:
        if k not in ignored_keys and k not in special_keys:
            vals = []
            for b in flat_batches:
                if 'task_neg_dst' in b.task_data and (k != 'task_neg_dst' and k != 'ts'):
                    continue
                if k not in b.task_data:
                    continue
                vals.append(b.task_data[k])
            #if(k == 'task_neg_dst'):
                #print(vals)
            if isinstance(vals[0], torch.Tensor):
                merged_task[k] = torch.cat(vals, dim=0)


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
            dt_list = []
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
                src_indices_list.append(torch.arange(b.layer_data[l-1]['gids'].shape[0]) if l > 0 else torch.arange(b.task_data['gids'].shape[0]) + current_gid_offset)
                #src_indices_list.append(torch.arange(len(layer['indices'])) + current_gid_offset)
                indices_list.append(layer['indices'] +  current_gid_offset)
                current_gid_offset += layer['gids'].shape[0]
                if 'eid' in layer: eid_list.append(layer['eid'])
                if 'ts' in layer: ts_list.append(layer['ts'])
                if 'edge_dt' in layer: dt_list.append(layer['edge_dt'])
            l_data['indptr'] = torch.cat(indptr_list)
            l_data['indices'] = torch.cat(indices_list)
            l_data['gids'] = torch.cat(gids_list)
            l_data['src_indices'] = torch.cat(src_indices_list)
            if eid_list: l_data['eid'] = torch.cat(eid_list)
            if ts_list: l_data['ts'] = torch.cat(ts_list)
            if dt_list: l_data['edge_dt'] = torch.cat(dt_list)
            
            
            merged_layers.append(l_data)
            batch_input_node_counts.append(layer_node_counts)

    # --- 4. Merge Route Plans (Re-indexing) ---
    merged_plans = []
    #ctx = DistributedContext.get_default_context()
    example_plans = flat_batches[0].comm_plans
    if example_plans is not None:
        for l in range(len(example_plans)):
            print(example_plans[l])
            send_sizes = torch.zeros(world_size, dtype=torch.long)
            send_locs = [[] for i in range(world_size)]
            remote_locs = [[] for i in range(world_size)]
            send_ranks = [[] for i in range(world_size)]
            for b_idx, b in enumerate(flat_batches):
                if l is None or l >= len(b.comm_plans): continue
                plan:CommPlan = b.comm_plans[l]
                if plan is None: continue
                curr_rank_start = 0
                send_sizes += plan.send_sizes
                for r in range(world_size):
                    curr_rank_end = curr_rank_start + plan.send_sizes[r]
                    send_locs[r].append(plan.send_indices[curr_rank_start:curr_rank_end])
                    #send_locs[r].extend(new_send_locs[new_send_locs[curr_rank_start:curr_rank_end]])
                    remote_locs[r].append(plan.send_remote_indices[curr_rank_start:curr_rank_end])
                    send_ranks[r].append(plan.send_ranks[curr_rank_start:curr_rank_end])
                    curr_rank_start = curr_rank_end
            new_ranks = [torch.cat(send_ranks[r]) for r in range(world_size)]
            new_indices = [torch.cat(send_locs[r]) for r in range(world_size)]
            new_remote_indices = [torch.cat(remote_locs[r]) for r in range(world_size)]
            new_ranks = torch.cat(new_ranks)
            new_indices = torch.cat(new_indices)
            new_remote_indices = torch.cat(new_remote_indices)
            merged_plans.append(CommPlan(new_ranks, send_sizes, new_indices, new_remote_indices))
    return MergedBatch(merged_task, merged_layers, merged_plans)


