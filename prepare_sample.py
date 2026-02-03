import sys
import os
from pathlib import Path
import gc
from typing import Dict, List, Tuple, Optional, Union
from functools import partial
import multiprocessing as mp
import time

# [关键设置] 必须在导入 numpy/numba 前设置，防止多进程死锁
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 自动添加项目根目录到环境变量
current_file = Path(__file__).resolve()
project_root = current_file.parents[2] 
sys.path.append(str(project_root))

import torch
import numpy as np
from tqdm import tqdm
import numba

# 尝试导入 CommPlan，如果没有则定义一个简单的容器类
try:
    from starrygl.cache.cache_route import CommPlan
except ImportError:
    class CommPlan:
        def __init__(self, send_ranks, send_sizes, send_indices, send_remote_indices):
            self.send_ranks = send_ranks
            self.send_sizes = send_sizes
            self.send_indices = send_indices
            self.send_remote_indices = send_remote_indices

# ==============================================================================
# Part 1: 高性能 Numba 算子 (核心加速部分)
# ==============================================================================

@numba.njit(fastmath=True)
def _jit_get_unique_pairs_map(nodes, ts):
    """
    [加速] 使用 Hash Map 实现 O(N) 去重，替代 np.unique (O(N log N))
    """
    n = len(nodes)
    unique_ids = np.empty(n, dtype=nodes.dtype)
    unique_ts = np.empty(n, dtype=ts.dtype)
    inverse = np.empty(n, dtype=np.int64)
    
    # Numba Dict: Key=(node_id, ts), Value=new_index
    mapping = {} 
    count = 0
    
    for i in range(n):
        u = nodes[i]
        t = ts[i]
        key = (u, t)
        
        if key in mapping:
            inverse[i] = mapping[key]
        else:
            mapping[key] = count
            unique_ids[count] = u
            unique_ts[count] = t
            inverse[i] = count
            count += 1
            
    return unique_ids[:count], unique_ts[:count], inverse

@numba.njit(fastmath=True)
def _jit_merge_and_map(curr_ids, curr_ts, raw_src, raw_ts):
    """
    [加速] 合并上一层节点和新采样节点，生成下一层的映射表
    """
    n_curr = len(curr_ids)
    n_raw = len(raw_src)
    total_max = n_curr + n_raw
    
    out_ids = np.empty(total_max, dtype=curr_ids.dtype)
    out_ts = np.empty(total_max, dtype=curr_ts.dtype)
    raw_inverse = np.empty(n_raw, dtype=np.int64)
    
    mapping = {}
    count = 0
    
    # 1. 优先保留 curr_ids (Target Nodes)
    for i in range(n_curr):
        u = curr_ids[i]
        t = curr_ts[i]
        key = (u, t)
        if key not in mapping:
            mapping[key] = count
            out_ids[count] = u
            out_ts[count] = t
            count += 1
            
    # 2. 加入新采样的 raw_src 并记录映射
    for i in range(n_raw):
        u = raw_src[i]
        t = raw_ts[i]
        key = (u, t)
        if key in mapping:
            raw_inverse[i] = mapping[key]
        else:
            idx = count
            mapping[key] = idx
            out_ids[idx] = u
            out_ts[idx] = t
            raw_inverse[i] = idx
            count += 1
            
    return out_ids[:count], out_ts[:count], raw_inverse

@numba.njit(fastmath=True)
def _jit_build_csc_bucket(src_indices, dst_indices, ts, dt, eid, num_col_nodes):
    """
    [加速] 使用桶排序思想构建 CSC，避免全局排序
    """
    num_edges = len(src_indices)
    indptr = np.zeros(num_col_nodes + 1, dtype=np.int64)
    
    # 1. 计算度数
    for i in range(num_edges):
        col = dst_indices[i]
        indptr[col + 1] += 1
        
    # 2. 前缀和
    for i in range(num_col_nodes):
        indptr[i+1] += indptr[i]
        
    # 3. 填充数据 (保持相对顺序，或依赖 GNN 对顺序不敏感)
    out_src = np.empty(num_edges, dtype=src_indices.dtype)
    out_ts = np.empty(num_edges, dtype=ts.dtype)
    out_dt = np.empty(num_edges, dtype=dt.dtype)
    out_eid = np.empty(num_edges, dtype=eid.dtype)
    
    temp_pos = indptr.copy()
    for i in range(num_edges):
        col = dst_indices[i]
        pos = temp_pos[col]
        out_src[pos] = src_indices[i]
        out_ts[pos] = ts[i]
        out_dt[pos] = dt[i]
        out_eid[pos] = eid[i]
        temp_pos[col] += 1
        
    return indptr, out_src, out_ts, out_dt, out_eid

@numba.njit(fastmath=True)
def _jit_compute_route_cpu(unique_gids, u_ts, node_parts, pid):
    """
    [加速] 纯 CPU 路由计算，替代 torch_scatter
    """
    n = len(unique_gids)
    # Map: gid -> (max_ts, original_index)
    best_ts = {}
    best_idx = {}
    
    # 1. 找到每个 GID 对应的最大时间戳
    for i in range(n):
        gid = unique_gids[i]
        t = u_ts[i]
        if gid in best_ts:
            if t > best_ts[gid]:
                best_ts[gid] = t
                best_idx[gid] = i
        else:
            best_ts[gid] = t
            best_idx[gid] = i
            
    # 2. 筛选属于当前 pid (Master) 的节点
    out_gids_list = []
    out_indices_list = []
    
    # Numba 不支持直接遍历 dict.items()，需遍历 keys
    # 注意：dict keys 顺序在 python 3.7+ 保留，但在 Numba 中可能未定，但这不影响正确性
    for gid in best_idx:
        # 边界检查
        if gid < len(node_parts):
            owner = node_parts[gid]
            if owner == pid:
                out_gids_list.append(gid)
                out_indices_list.append(best_idx[gid])
            
    return np.array(out_gids_list, dtype=np.int64), np.array(out_indices_list, dtype=np.int64)

@numba.njit(parallel=False, fastmath=True)
def _jit_sample_kernel(
    active_starts, active_ends, active_ts, active_indices, 
    sorted_src, sorted_ts, sorted_eid, sorted_cluster,     
    fanout, sample_type, graph_type,                       
    start_offset, end_offset, cluster_id_val               
):
    """
    核心采样逻辑
    """
    num_tasks = len(active_starts)
    counts = np.zeros(num_tasks, dtype=np.int64)
    
    # Pass 1: Count
    for i in range(num_tasks):
        s, e, t_cut = active_starts[i], active_ends[i], active_ts[i]
        nb_ts_view = sorted_ts[s:e]
        cut_idx = np.searchsorted(nb_ts_view, t_cut + end_offset)
        start_idx = np.searchsorted(nb_ts_view, t_cut + start_offset)
        
        if cut_idx > len(nb_ts_view): cut_idx = len(nb_ts_view)
        if start_idx < 0: start_idx = 0
        if start_idx > cut_idx: start_idx = cut_idx

        if cut_idx == 0:
            counts[i] = 0
            continue
            
        count = 0
        if graph_type == 0: # continuous
            if sample_type == 2: count = cut_idx 
            elif sample_type == 0 or sample_type == 1: count = min(cut_idx, fanout)
            else: count = cut_idx
        else: # discrete
            if sample_type == 2:
                c_match = 0
                for k in range(s + start_idx, s + cut_idx):
                    if sorted_cluster[k] == cluster_id_val: c_match += 1
                count = c_match
            elif sample_type == 1:
                valid_len = cut_idx - start_idx
                count = min(valid_len, fanout) if valid_len > 0 else 0
            else:
                count = max(0, cut_idx - start_idx)
        counts[i] = count

    total_edges = np.sum(counts)
    offsets = np.zeros(num_tasks + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    
    out_src = np.empty(total_edges, dtype=sorted_src.dtype)
    out_dst = np.empty(total_edges, dtype=active_indices.dtype)
    out_ts  = np.empty(total_edges, dtype=sorted_ts.dtype)
    out_dt  = np.empty(total_edges, dtype=np.float32) 
    out_eid = np.empty(total_edges, dtype=sorted_eid.dtype)
    
    # Pass 2: Fill
    for i in range(num_tasks):
        count = counts[i]
        if count == 0: continue
        write_start, target_id, t_cut = offsets[i], active_indices[i], active_ts[i]
        s, e = active_starts[i], active_ends[i]
        nb_ts_view = sorted_ts[s:e]
        cut_idx = np.searchsorted(nb_ts_view, t_cut + end_offset)
        start_idx = np.searchsorted(nb_ts_view, t_cut + start_offset)
        
        if cut_idx > len(nb_ts_view): cut_idx = len(nb_ts_view)
        if start_idx < 0: start_idx = 0
        if start_idx > cut_idx: start_idx = cut_idx
        
        current_ptr = write_start
        if graph_type == 0: # continuous
            real_start, real_end = s, s
            if sample_type == 0: # recent
                offset_back = max(0, cut_idx - fanout)
                real_start, real_end = s + offset_back, s + cut_idx
            elif sample_type == 1: # uniform
                if cut_idx <= fanout:
                    real_start, real_end = s, s + cut_idx
                else:
                    # 简单间隔采样
                    step = cut_idx // fanout
                    for k in range(fanout):
                        idx = s + k * step
                        if idx >= s + cut_idx: idx = s + cut_idx - 1
                        out_src[current_ptr] = sorted_src[idx]
                        out_ts[current_ptr]  = sorted_ts[idx]
                        out_eid[current_ptr] = sorted_eid[idx]
                        out_dst[current_ptr] = target_id
                        out_dt[current_ptr]  = t_cut - sorted_ts[idx]
                        current_ptr += 1
                    continue
            else: # full
                real_start, real_end = s, s + cut_idx
            
            w_len = real_end - real_start
            if w_len > 0:
                out_src[write_start : write_start + w_len] = sorted_src[real_start : real_end]
                out_ts[write_start : write_start + w_len]  = sorted_ts[real_start : real_end]
                out_eid[write_start : write_start + w_len] = sorted_eid[real_start : real_end]
                out_dst[write_start : write_start + w_len] = target_id 
                out_dt[write_start : write_start + w_len]  = t_cut - sorted_ts[real_start : real_end]
        else: # discrete
            added = 0
            for k in range(s + start_idx, s + cut_idx):
                if added >= count: break
                if sample_type == 2 and sorted_cluster[k] != cluster_id_val: continue
                out_src[current_ptr] = sorted_src[k]
                out_ts[current_ptr]  = t_cut 
                out_eid[current_ptr] = sorted_eid[k]
                out_dst[current_ptr] = target_id
                out_dt[current_ptr]  = t_cut - sorted_ts[k]
                current_ptr += 1
                added += 1

    return out_src, out_dst, out_ts, out_dt, out_eid

# ==============================================================================
# Part 2: Materialized Builder (完全集成了优化算子)
# ==============================================================================

class MaterializedMultiLayerBuilder:
    def __init__(self, partition_data, num_nodes, node_parts, rep_table, pid, num_parts):
        self.hist_src = partition_data['src'].astype(np.int64)
        self.hist_dst = partition_data['dst'].astype(np.int64)
        self.hist_ts = partition_data['ts'].astype(np.int64)
        self.hist_cluster = partition_data.get('cid', np.zeros_like(self.hist_src, dtype=np.int64))
        self.hist_eid = partition_data.get('eid', np.arange(len(self.hist_src), dtype=np.int64))
        self.num_nodes = num_nodes
        
        # [优化] 确保 node_parts 是 numpy array，以便传给 Numba
        if isinstance(node_parts, torch.Tensor):
            self.node_parts = node_parts.numpy()
        else:
            self.node_parts = node_parts
            
        self.rep_table = rep_table
        self.pid = pid
        self.num_parts = num_parts

        # 预排序
        sort_keys = np.lexsort((self.hist_ts, self.hist_dst))
        self.sorted_src = self.hist_src[sort_keys]
        self.sorted_dst = self.hist_dst[sort_keys]
        self.sorted_ts = self.hist_ts[sort_keys]
        self.sorted_eid = self.hist_eid[sort_keys]
        self.sorted_cluster = self.hist_cluster[sort_keys]
        
        # 构建 Full Indptr
        self.full_indptr = np.zeros(num_nodes + 1, dtype=np.int64)
        u, c = np.unique(self.sorted_dst, return_counts=True)
        valid_mask = u < num_nodes
        self.full_indptr[u[valid_mask]+1] = c[valid_mask]
        np.cumsum(self.full_indptr, out=self.full_indptr)

    def _sample_layer(self, target_nodes, target_ts, fanout=10, sample_type='recent', meta_config=None, cluster_ids=None):
        starts = self.full_indptr[target_nodes]
        ends = self.full_indptr[target_nodes + 1]
        valid_mask = (ends - starts) > 0
        
        active_idx = np.where(valid_mask)[0]
        if len(active_idx) == 0: return None, None, None, None, None
        
        s_type = {'recent':0, 'uniform':1, 'cluster':2, 'full':3}.get(sample_type, 0)
        g_type = 0 if meta_config.get('graph_type', 'c') == 'c' else 1
        cid = cluster_ids if cluster_ids is not None else -1
        
        return _jit_sample_kernel(
            starts[valid_mask], ends[valid_mask], target_ts[valid_mask], active_idx,
            self.sorted_src, self.sorted_ts, self.sorted_eid, self.sorted_cluster,
            int(fanout), int(s_type), int(g_type),
            int(meta_config.get('start_offset', 0)), int(meta_config.get('end_offset', 0)), int(cid)
        )

    def _compute_layer_route_fast(self, layer_nodes, layer_ts):
        """
        使用 Numba 加速的路由计算，避免 Torch/Numpy 互转
        """
        if len(layer_nodes) == 0: return None
        
        # 1. 纯 CPU 筛选 Master 节点 (O(N))
        query_gids, query_original_indices = _jit_compute_route_cpu(
            layer_nodes, layer_ts, self.node_parts, self.pid
        )
        
        if len(query_gids) == 0: return None

        # 2. 查表 (Lookup) - 这里必须用 Torch 因为 rep_table 是 Torch 优化的
        t_query_gids = torch.from_numpy(query_gids)
        src_indices_in_query, target_ranks, target_locs = self.rep_table.lookup(t_query_gids)
        
        # 3. 过滤本地请求
        owner_filter = target_ranks != self.pid
        if not owner_filter.any(): return None

        src_indices_in_query = src_indices_in_query[owner_filter]
        target_ranks = target_ranks[owner_filter]
        target_locs = target_locs[owner_filter]
        
        # 映射回本地索引
        final_local_indices = torch.from_numpy(query_original_indices)[src_indices_in_query]
        
        # 排序并打包
        sort_idx = torch.argsort(target_ranks)
        return CommPlan(
            send_ranks=target_ranks[sort_idx],
            send_sizes=None,
            send_indices=final_local_indices[sort_idx],
            send_remote_indices=target_locs[sort_idx]
        )

    def compress_tensor(self, x):
        """Int64 -> Int32 压缩"""
        if not isinstance(x, torch.Tensor): return x
        if x.dtype == torch.int64:
            if x.numel() > 0 and x.max() < 2147483647 and x.min() > -2147483648:
                return x.to(torch.int32)
        elif x.dtype == torch.float64:
            return x.to(torch.float32)
        return x

    def build_batch_in_memory(self, task_param, layer_configs, num_neg=1, global_dst_pool=None, cluster_ids=None):
        task_type, task_nodes = task_param
        
        if task_type == 'link':
            task_src, task_dst, task_ts, task_label, task_eid = task_nodes 
            l0_nodes = np.concatenate([task_src, task_dst])
            l0_ts = np.concatenate([task_ts, task_ts])
        elif task_type == 'node': 
            task_node, task_ts, task_label = task_nodes
            l0_nodes, l0_ts = task_node, task_ts
            neg_dst_pool = None
        elif task_type == 'neg':
            if num_neg > 0 and global_dst_pool is not None:
                task_ts = task_nodes
                rand_indices = np.random.randint(0, len(global_dst_pool), (len(task_ts), num_neg)).reshape(-1)
                neg_dst_pool = global_dst_pool[rand_indices]
                l0_nodes = neg_dst_pool.flatten()
                l0_ts = np.repeat(task_ts, num_neg)
            else: return None
        else: return None

        # Step B: L0 Unique (使用 Numba Hash Map)
        curr_uid, curr_uts, l0_inv = _jit_get_unique_pairs_map(l0_nodes, l0_ts)
        batch_data = [{"gids": curr_uid, "ts": curr_uts, "inv_map": l0_inv, "num_neg": num_neg}]
        layer_routes = []
        
        # Route 0
        r0 = self._compute_layer_route_fast(curr_uid, curr_uts)
        layer_routes.append(r0)
        
        # Step C: Multi-hop Sampling
        last_inv_map = None
        for l, config in enumerate(layer_configs['layer']):
            target_u = curr_uid if l==0 else curr_uid[last_inv_map]
            target_t = curr_uts if l==0 else curr_uts[last_inv_map]
            
            raw_src, raw_dst, raw_ts, raw_dt, raw_eid = self._sample_layer(
                target_u, target_t, config['fanout'], config['type'], layer_configs['meta'], cluster_ids
            )
            
            if raw_src is None:
                # Empty layer handling
                batch_data.append({
                    "indptr": np.zeros(1, dtype=np.int64), "indices": np.array([], dtype=np.int64),
                    "eid": np.array([], dtype=np.int64), "edge_ts": np.array([], dtype=np.int64),
                    "edge_dt": np.array([], dtype=np.float32), "gids": np.array([], dtype=np.int64), "ts": np.array([], dtype=np.int64)
                })
                layer_routes.append(None)
                break
                
            raw_dst_mapped = last_inv_map[raw_dst] if last_inv_map is not None else raw_dst
            
            # Merge & Map (使用 Numba)
            next_uid, next_uts, next_inv = _jit_merge_and_map(curr_uid, curr_uts, raw_src, raw_ts)
            
            # Build CSC (使用 Numba 桶排序)
            ptr, idx, b_ts, b_dt, b_eid = _jit_build_csc_bucket(
                next_inv, raw_dst_mapped, raw_ts, raw_dt, raw_eid, curr_uid.shape[0]
            )
   
            batch_data.append({
                "indptr": ptr, "indices": idx, "eid": b_eid, "edge_ts": b_ts,
                "edge_dt": b_dt, "gids": next_uid, "ts": next_uts
            })
            
            if l < len(layer_configs['layer']):
                r_next = self._compute_layer_route_fast(next_uid, next_uts)
                layer_routes.append(r_next)
                
            curr_uid, curr_uts, last_inv_map = next_uid, next_uts, next_inv
            
        # Final Conversion and Compression
        def to_tensor(x): return torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        
        final_data = []
        for d in batch_data:
            compressed_d = {k: self.compress_tensor(to_tensor(v)) for k,v in d.items()}
            final_data.append(compressed_d)
            
        if task_type == 'link':
            final_data[0].update({
                "task_ts": self.compress_tensor(to_tensor(task_ts)), 
                "task_src": self.compress_tensor(to_tensor(task_src)),
                "task_dst": self.compress_tensor(to_tensor(task_dst)), 
                "task_label": self.compress_tensor(to_tensor(task_label)), 
                "task_eid": self.compress_tensor(to_tensor(task_eid))
            })
        elif task_type == 'node':
            final_data[0].update({
                "task_node": self.compress_tensor(to_tensor(task_node)), 
                "task_ts": self.compress_tensor(to_tensor(task_ts)), 
                "task_label": self.compress_tensor(to_tensor(task_label))
            })
        elif task_type == 'neg':
            final_data[0].update({
                "task_nodes": self.compress_tensor(to_tensor(neg_dst_pool)), 
                "task_ts": self.compress_tensor(to_tensor(task_ts)),
                'task_label': 'neg_set'
            })
        
        if layer_routes:
            for r in layer_routes:
                if r is not None:
                    r.send_indices = self.compress_tensor(r.send_indices)
                    r.send_remote_indices = self.compress_tensor(r.send_remote_indices)
        
        final_data[0]['comm_plan'] = layer_routes
        return final_data

# ==============================================================================
# Part 3: Worker & Main Flow
# ==============================================================================

def worker_process_slot(c_path, builder_instance, layer_configs, num_neg, global_dst_pool, task_type, global_edge_label, global_node_label, num_set):
    results = []
    try:
        c_data = torch.load(c_path, map_location='cpu')
        sub_id = int(c_path.stem.split('_')[3])
        
        if task_type == 'link':
            task_src = c_data['src'].numpy().astype(np.int64)
            task_dst = c_data['dst'].numpy().astype(np.int64)
            task_ts = c_data['ts'].numpy().astype(np.int64)
            task_eid = c_data['eid'].numpy().astype(np.int64)
            
            if global_edge_label is not None:
                labels = global_edge_label[task_eid]
                if isinstance(labels, torch.Tensor): labels = labels.numpy()
            else:
                labels = np.ones_like(task_src, dtype=np.int64)
            
            task_param = ('link', (task_src, task_dst, task_ts, labels, task_eid))
        else:
            raw_n, raw_t = c_data['src'].numpy(), c_data['ts'].numpy()
            node, idx = np.unique(raw_n, return_index=True)
            node_ts = raw_t[idx]
            
            labels = np.zeros_like(node, dtype=np.int64)
            if global_node_label is not None:
                if isinstance(global_node_label, torch.Tensor):
                    labels = global_node_label[node].numpy()
            
            task_param = ('node', (node, node_ts, labels))
        
        # Sample Main
        main_res = builder_instance.build_batch_in_memory(
            task_param, layer_configs, num_neg, global_dst_pool, sub_id
        )
        if main_res: results.append(main_res)
        
        # Sample Neg
        if task_type == 'link' and num_set > 0:
            for i in range(num_set):
                neg_param = ('neg', task_ts)
                neg_res = builder_instance.build_batch_in_memory(
                    neg_param, layer_configs, num_neg=1, global_dst_pool=global_dst_pool, cluster_ids=sub_id
                )
                if neg_res: results.append(neg_res)
                    
        return results
    except Exception as e:
        return f"Error in {c_path.name}: {str(e)}"

def run_offline_mega_batch(
    raw_chunk_dir, output_root, num_nodes, layer_configs, 
    num_set=8, num_neg=1, task_type='link', add_inverse=True,
    global_node_label=None, global_edge_label=None,
    node_parts=None, rep_table=None, pid=0, num_parts=1, partition_book=None
):
    print(f"\n[Step 6] Running Optimized Mega-Batch Materialization...")
    output_root.mkdir(parents=True, exist_ok=True)
    
    # [Tuning] 根据内存调整，建议 10-20
    PACK_SIZE = 10 
    # [Tuning] 建议设置为 4-8，不要占满所有核，留给 IO
    NUM_WORKERS = 4 

    global_dst_pool_all = None
    if num_neg > 0:
        print("Building global dst pool...")
        # 简单构建 pool 逻辑
        dst_list = []
        for f in raw_chunk_dir.glob("part_*/slot_*.pt"):
            try: dst_list.append(torch.load(f)['dst'].numpy())
            except: pass
        if dst_list:
            global_dst_pool_all = np.unique(np.concatenate(dst_list))

    part_dirs = sorted([d for d in raw_chunk_dir.glob("part_*") if d.is_dir()])
    
    for p_dir in part_dirs:
        curr_pid = int(p_dir.name.split('_')[1])
        print(f"  > Processing Part {curr_pid}...")
        save_dir = output_root / p_dir.name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Local History
        all_chunks = sorted(list(p_dir.glob("slot_*.pt")), key=lambda x: x.name)
        if not all_chunks: continue
        
        hist_src, hist_dst, hist_ts, hist_eid, hist_cid = [], [], [], [], []
        for c_path in tqdm(all_chunks, desc="Loading History"):
            try:
                sub_id = int(c_path.stem.split('_')[3])
                c_data = torch.load(c_path) 
                s, d, t, e = c_data['src'].numpy(), c_data['dst'].numpy(), c_data['ts'].numpy(), c_data['eid'].numpy()
                hist_src.append(s); hist_dst.append(d); hist_ts.append(t); hist_eid.append(e)
                hist_cid.append(np.full_like(s, sub_id))
                if add_inverse:
                    hist_src.append(d); hist_dst.append(s); hist_ts.append(t); hist_eid.append(e)
                    hist_cid.append(np.full_like(d, sub_id))
            except: pass

        if not hist_src: continue
        local_data = {
            'src': np.concatenate(hist_src), 'dst': np.concatenate(hist_dst),
            'ts': np.concatenate(hist_ts), 'eid': np.concatenate(hist_eid),
            'cid': np.concatenate(hist_cid)
        }
        del hist_src, hist_dst, hist_ts, hist_eid, hist_cid
        gc.collect()

        builder = MaterializedMultiLayerBuilder(
            local_data, num_nodes, node_parts=node_parts, 
            rep_table=rep_table, pid=curr_pid, num_parts=num_parts
        )
        
        local_dst_pool = None
        if global_dst_pool_all is not None:
             local_dst_pool = global_dst_pool_all[node_parts[global_dst_pool_all] == curr_pid]

        chunked_files = [all_chunks[i:i + PACK_SIZE] for i in range(0, len(all_chunks), PACK_SIZE)]
        
        batch_idx = 0
        worker_func = partial(
            worker_process_slot,
            builder_instance=builder, layer_configs=layer_configs,
            num_neg=num_neg, global_dst_pool=local_dst_pool,
            task_type=task_type, global_edge_label=global_edge_label,
            global_node_label=global_node_label, num_set=num_set
        )
        
        for files_batch in tqdm(chunked_files, desc="Mega-Batches"):
            mega_batch_results = []
            with mp.Pool(processes=NUM_WORKERS) as pool:
                raw_results = pool.map(worker_func, files_batch)
                for r in raw_results:
                    if isinstance(r, str): print(f"Warning: {r}")
                    else: mega_batch_results.extend(r)
            
            if mega_batch_results:
                torch.save(mega_batch_results, save_dir / f"mega_batch_{batch_idx}.pt")
                batch_idx += 1
            gc.collect()
            
        del local_data, builder
        gc.collect()

if __name__ == "__main__":
    # 配置路径
    src_root = Path("/mnt/data/zlj/starrygl-data/ctdg").resolve()
    tgt_root = Path("/mnt/data/zlj/starrygl-data/nparts").resolve()
    processed_root = Path("/mnt/data/zlj/starrygl-data/processed_atomic").resolve()
    
    num_inter_parts = 4
    LAYER_CONFIGS = {
        'meta': {'start_offset': -1, 'end_offset': 0, 'graph_type': 'c'},
        'layer': [{'type': 'recent', 'fanout': 10}, {'type': 'recent', 'fanout': 10}]
    }

    for p in src_root.glob("*.pth"):
        name = p.stem
        if name == 'BCB': continue # Skip large if needed
        if name != 'WikiTalk': continue # 仅处理指定数据集，调试用
        print(f"Processing {name}...")
        
        raw_data_dict = torch.load(p, map_location='cpu')
        num_nodes = raw_data_dict['num_nodes']
        dataset = raw_data_dict['dataset']
        
        node_label, edge_label = None, None
        perm_path = tgt_root / f"{name}_{num_inter_parts:03d}"/"perm.pt"
        
        if perm_path.exists():
            perm, _ = torch.load(perm_path)
            if isinstance(dataset, dict):
                if 'node_label' in dataset: node_label = dataset['node_label'][perm]
                elif 'y' in dataset: node_label = dataset['y'][perm]
                if 'edge_label' in dataset: edge_label = dataset['edge_label']
            elif isinstance(dataset, list):
                if 'edge_label' in dataset[0]:
                    edge_label = torch.cat([d['edge_label'] for d in dataset])  
        
        del raw_data_dict
        
        raw_dir = tgt_root / f"{name}_{num_inter_parts:03d}"
        if not raw_dir.exists(): continue
            
        rep_table = torch.load(raw_dir / "replica_table.pt")
        partition_book_data = torch.load(raw_dir / "partition_book.pt")
        node_parts = partition_book_data[1] # Must be numpy array!
        if isinstance(node_parts, torch.Tensor): node_parts = node_parts.numpy()

        final_output_dir = processed_root / f"{name}_{num_inter_parts:03d}"
        
        run_offline_mega_batch(
            raw_dir, final_output_dir,
            num_nodes, LAYER_CONFIGS, num_neg=1, add_inverse=True,
            global_node_label=node_label, global_edge_label=edge_label,
            node_parts=node_parts, rep_table=rep_table, pid=0, 
            num_parts=num_inter_parts
        )
        print("Done.")