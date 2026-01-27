import sys
import os
from pathlib import Path
import gc
from typing import Dict, List, Tuple, Optional, Union

import torch_scatter
os.environ["NUMBA_NUM_THREADS"] = "32"
# [Fix 1] 自动添加项目根目录到环境变量，解决 ModuleNotFoundError
current_file = Path(__file__).resolve()
# 假设结构为 Project/starrygl/graph/prepare_sample.py，根目录在 parents[2]
project_root = current_file.parents[2] 
sys.path.append(str(project_root))

import torch
import numpy as np
import dgl
from tqdm import tqdm
import numba

# 尝试导入 CommPlan 以确保反序列化正常
try:
    from starrygl.cache.cache_route import CommPlan
except ImportError:
    pass

# ==============================================================================
# Part 1: Numba JIT Sampling Kernel (高效采样内核)
# ==============================================================================


@numba.njit(parallel=True, fastmath=True)
def _jit_sample_kernel(
    active_starts, active_ends, active_ts, active_indices, 
    sorted_src, sorted_ts, sorted_eid, sorted_cluster,     
    fanout, sample_type, graph_type,                       
    start_offset, end_offset, cluster_id_val               
):
    num_tasks = len(active_starts)
    counts = np.zeros(num_tasks, dtype=np.int64)
    
    # --- Pass 1: Count Edges ---
    for i in numba.prange(num_tasks):
        s = active_starts[i]
        e = active_ends[i]
        t_cut = active_ts[i]
        
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
                    if sorted_cluster[k] == cluster_id_val:
                        c_match += 1
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
    
    # --- Pass 2: Fill Data ---
    for i in numba.prange(num_tasks):
        count = counts[i]
        if count == 0: continue
            
        write_start = offsets[i]
        target_id = active_indices[i]
        t_cut = active_ts[i]
        
        s = active_starts[i]
        e = active_ends[i]
        
        nb_ts_view = sorted_ts[s:e]
        cut_idx = np.searchsorted(nb_ts_view, t_cut + end_offset)
        start_idx = np.searchsorted(nb_ts_view, t_cut + start_offset)
        
        if cut_idx > len(nb_ts_view): cut_idx = len(nb_ts_view)
        if start_idx < 0: start_idx = 0
        if start_idx > cut_idx: start_idx = cut_idx
        
        current_ptr = write_start
        
        if graph_type == 0: # continuous
            real_start, real_end = s, s
            
            if sample_type == 2: # cluster
                real_start, real_end = s, s + cut_idx
            elif sample_type == 0: # recent
                offset_back = max(0, cut_idx - fanout)
                real_start, real_end = s + offset_back, s + cut_idx
            elif sample_type == 1: # uniform
                if cut_idx <= fanout:
                    real_start, real_end = s, s + cut_idx
                else:
                    rand_idxs = np.random.choice(cut_idx, fanout, replace=False)
                    rand_idxs.sort() 
                    for k in range(fanout):
                        idx = s + rand_idxs[k]
                        out_src[current_ptr] = sorted_src[idx]
                        out_ts[current_ptr]  = sorted_ts[idx]
                        out_eid[current_ptr] = sorted_eid[idx]
                        out_dst[current_ptr] = target_id
                        out_dt[current_ptr]  = t_cut - sorted_ts[idx]
                        current_ptr += 1
                    continue 
            else: # full
                real_start, real_end = s, s + cut_idx
            
            if current_ptr == write_start: 
                w_len = real_end - real_start
                if w_len > 0:
                    out_src[write_start : write_start + w_len] = sorted_src[real_start : real_end]
                    out_ts[write_start : write_start + w_len]  = sorted_ts[real_start : real_end]
                    out_eid[write_start : write_start + w_len] = sorted_eid[real_start : real_end]
                    out_dst[write_start : write_start + w_len] = target_id 
                    out_dt[write_start : write_start + w_len]  = t_cut - sorted_ts[real_start : real_end]
        
        else: # discrete
            if sample_type == 2: 
                added = 0
                for k in range(s + start_idx, s + cut_idx):
                    if sorted_cluster[k] == cluster_id_val:
                        if added < count:
                            out_src[current_ptr] = sorted_src[k]
                            out_ts[current_ptr]  = t_cut 
                            out_eid[current_ptr] = sorted_eid[k]
                            out_dst[current_ptr] = target_id
                            out_dt[current_ptr]  = t_cut - sorted_ts[k]
                            current_ptr += 1
                            added += 1
            elif sample_type == 3 or sample_type == 1:
                 real_start = s + start_idx
                 real_end = s + cut_idx
                 w_len = real_end - real_start
                 if w_len > 0 and count > 0:
                     copy_len = min(w_len, count)
                     out_src[write_start:write_start+copy_len] = sorted_src[real_start:real_start+copy_len]
                     out_ts[write_start:write_start+copy_len] = t_cut
                     out_eid[write_start:write_start+copy_len] = sorted_eid[real_start:real_start+copy_len]
                     out_dst[write_start:write_start+copy_len] = target_id
                     out_dt[write_start:write_start+copy_len] = t_cut - sorted_ts[real_start:real_start+copy_len]

    return out_src, out_dst, out_ts, out_dt, out_eid

# ==============================================================================
# Part 2: Materialized Builder
# ==============================================================================
    
class MaterializedMultiLayerBuilder:
    def __init__(self, partition_data, num_nodes, node_parts, rep_table, pid, num_parts):
        self.hist_src = partition_data['src']
        self.hist_dst = partition_data['dst']
        self.hist_ts = partition_data['ts']
        self.hist_cluster = partition_data.get('cid', np.zeros_like(self.hist_src, dtype=np.int64))
        self.hist_eid = partition_data.get('eid', np.arange(len(self.hist_src), dtype=np.int64))
        self.num_nodes = num_nodes
        
        # [New] Routing context
        self.node_parts = node_parts # Global NewID -> PartID
        self.rep_table = rep_table
        self.pid = pid
        self.num_parts = num_parts
        sort_keys = np.lexsort((self.hist_ts, self.hist_dst))
        self.sorted_src = self.hist_src[sort_keys]
        self.sorted_dst = self.hist_dst[sort_keys]
        self.sorted_ts = self.hist_ts[sort_keys]
        self.sorted_eid = self.hist_eid[sort_keys]
        self.sorted_cluster = self.hist_cluster[sort_keys]
        
        self.full_indptr = np.zeros(num_nodes + 1, dtype=np.int64)
        u, c = np.unique(self.sorted_dst, return_counts=True)
        valid_mask = u < num_nodes
        self.full_indptr[u[valid_mask]+1] = c[valid_mask]
        np.cumsum(self.full_indptr, out=self.full_indptr)

    def _get_unique_pairs(self, nodes, timestamps):
        dt = np.dtype([('id', nodes.dtype), ('ts', timestamps.dtype)])
        combined = np.empty(len(nodes), dtype=dt)
        combined['id'] = nodes
        combined['ts'] = timestamps
        u_combined, inverse = np.unique(combined, return_inverse=True)
        return u_combined['id'], u_combined['ts'], inverse

    def _build_csc_from_pairs(self, src_local, dst_local, ts, dt, eid, num_dst_nodes):
        if len(src_local) == 0:
            return (np.zeros(num_dst_nodes + 1, dtype=np.int64), np.array([], dtype=np.int64), 
                    np.array([], dtype=np.int64), np.array([], dtype=np.float32), np.array([], dtype=np.int64))

        perm = np.lexsort((ts, dst_local))
        s_src, s_dst, s_ts, s_dt, s_eid = src_local[perm], dst_local[perm], ts[perm], dt[perm], eid[perm]
        
        indptr = np.zeros(num_dst_nodes + 1, dtype=np.int64)
        u, c = np.unique(s_dst, return_counts=True)
        valid_mask = u < num_dst_nodes
        indptr[u[valid_mask]+1] = c[valid_mask]
        np.cumsum(indptr, out=indptr)
        
        return indptr, s_src, s_ts, s_dt, s_eid

    def _merge_and_map_nodes(self, curr_ids, curr_ts, raw_ids, raw_ts):
        dt = np.dtype([('id', curr_ids.dtype), ('ts', curr_ts.dtype)])
        curr_struct = np.empty(len(curr_ids), dtype=dt)
        curr_struct['id'] = curr_ids; curr_struct['ts'] = curr_ts
        
        raw_struct = np.empty(len(raw_ids), dtype=dt)
        raw_struct['id'] = raw_ids; raw_struct['ts'] = raw_ts
        
        # Merge all unique
        all_struct = np.concatenate([curr_struct, raw_struct])
        unique_struct = np.unique(all_struct) # Sorted and Unique
        
        # Map raw_struct to unique_struct indices
        final_inverse = np.searchsorted(unique_struct, raw_struct)
        
        return unique_struct['id'], unique_struct['ts'], final_inverse
    
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
    def _compute_layer_route(self, layer_nodes, layer_ts):
        """
        Compute route for a specific layer's nodes.
        layer_nodes: Global IDs of nodes needed at this layer.
        """
        if len(layer_nodes) == 0:
            return None
            
        u_nodes = torch.from_numpy(layer_nodes) if isinstance(layer_nodes, np.ndarray) else layer_nodes
        u_ts = torch.from_numpy(layer_ts) if isinstance(layer_ts, np.ndarray) else layer_ts
        unique_gids, inverse_indices = torch.unique(u_nodes, return_inverse=True) 
        max_ts_val, argmax_indices = torch_scatter.scatter_max(u_ts, inverse_indices, dim=0)
        owners = self.node_parts[unique_gids]
        master_mask = (owners == self.pid)
        query_gids = unique_gids[master_mask]
        query_original_indices = argmax_indices[master_mask]
        if len(query_gids) == 0:
            return None
            
        # Lookup: Who needs my nodes?
        src_indices_in_query, target_ranks, target_locs = self.rep_table.lookup(query_gids)
        owner_filter = target_ranks != self.pid
        src_indices_in_query = src_indices_in_query[owner_filter]
        target_ranks = target_ranks[owner_filter]
        target_locs = target_locs[owner_filter]
        if len(target_ranks) == 0:
            return None
        
        final_local_indices = query_original_indices[src_indices_in_query]
        sort_idx = torch.argsort(target_ranks)
        final_ranks = target_ranks[sort_idx]
        final_local = final_local_indices[sort_idx]
        final_remote = target_locs[sort_idx]
        
        
        u_ranks, counts = torch.unique(final_ranks, return_counts=True)
        send_sizes = torch.zeros(self.num_parts, dtype=torch.long)
        send_sizes[u_ranks.long()] = counts.long()
        
        return CommPlan(
            send_ranks=final_ranks,
            send_sizes=send_sizes,
            send_indices=final_local,
            send_remote_indices=final_remote
        )


    def build_aligned_batch(self, task_param, save_path, layer_configs, num_neg=1, global_dst_pool=None, cluster_ids=None):
        task_type, task_nodes = task_param
        
        if task_type == 'link':
            task_src, task_dst, task_ts, task_label, task_eid = task_nodes 
            l0_nodes = np.concatenate([task_src, task_dst])
            l0_ts = np.concatenate([task_ts, task_ts])
        elif task_type == 'node': # node
            task_node, task_ts, task_label = task_nodes
            l0_nodes, l0_ts = task_node, task_ts
            neg_dst_pool = None
            
        elif task_type == 'neg':
            if num_neg > 0:
                if global_dst_pool is not None:
                    task_ts = task_nodes
                    rand_indices = np.random.randint(0, len(global_dst_pool), (len(task_ts), num_neg)).reshape(-1)
                    neg_dst_pool = global_dst_pool[rand_indices]
                    l0_nodes = neg_dst_pool.flatten()
                    l0_ts = np.repeat(task_ts, num_neg)
        else :
            raise ValueError(f"Unsupported task type: {task_type}")

        # Step B: L0 Unique
        curr_uid, curr_uts, l0_inv = self._get_unique_pairs(l0_nodes, l0_ts)
        batch_data = [{"gids": curr_uid, "ts": curr_uts, "inv_map": l0_inv, "num_neg": num_neg}]
        layer_routes = []
        r0 = self._compute_layer_route(curr_uid, curr_uts)
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
                curr_uid = np.array([], dtype=curr_uid.dtype)
                curr_uts = np.array([], dtype=curr_uts.dtype)
                batch_data.append({
                    "indptr": np.zeros(1, dtype=np.int64), "indices": np.array([], dtype=np.int64),
                    "eid": np.array([], dtype=np.int64), "edge_ts": np.array([], dtype=curr_uts.dtype),
                    "edge_dt": np.array([], dtype=np.float32), "gids": curr_uid, "ts": curr_uts
                })
                layer_routes.append(None)
                break
                
            raw_dst_mapped = last_inv_map[raw_dst] if last_inv_map is not None else raw_dst
            next_uid, next_uts, next_inv = self._merge_and_map_nodes(curr_uid, curr_uts, raw_src, raw_ts)
            ptr, idx, b_ts, b_dt, b_eid = self._build_csc_from_pairs(
                next_inv, raw_dst_mapped, raw_ts, raw_dt, raw_eid, curr_uid.shape[0]
            )
   
            batch_data.append({
                "indptr": ptr, "indices": idx, "eid": b_eid, "edge_ts": b_ts,
                "edge_dt": b_dt, "gids": next_uid, "ts": next_uts
            })
            if l < len(layer_configs['layer']):
                # 计算下一层的路由
                r_next = self._compute_layer_route(next_uid, next_uts)
                layer_routes.append(r_next)
            curr_uid, curr_uts, last_inv_map = next_uid, next_uts, next_inv
        ##print(layer_routes)
        # Save Logic
        def to_tensor(x): return torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        
        final_data = []
        for d in batch_data:
            final_data.append({k: to_tensor(v) for k,v in d.items()})
            
        if task_type == 'link':
            final_data[0].update({
                "task_ts": to_tensor(task_ts), "task_src": to_tensor(task_src),
                "task_dst": to_tensor(task_dst), 
                "task_label": to_tensor(task_label), "task_eid": to_tensor(task_eid) # [Saved with Reordered Label]
            })
            #if neg_dst_pool is not None: final_data[0]["neg_pool"] = to_tensor(neg_dst_pool)
        elif task_type == 'node':
            final_data[0].update({
                "task_node": to_tensor(task_node), "task_ts": to_tensor(task_ts), 
                "task_label": to_tensor(task_label) # [Saved with Reordered Label]
            })
        elif task_type == 'neg':
            final_data[0].update({
                "task_nodes": to_tensor(neg_dst_pool), "task_ts": to_tensor(task_ts),
                'task_label': 'neg_set' # [Saved with Reordered Label]
            })
        
        # Save Route (CommPlan) into the same file
        final_data[0]['comm_plan'] = layer_routes
        torch.save(final_data, save_path)

# ==============================================================================
# Part 3: Main Execution
# ==============================================================================

def build_global_dst_pool(raw_chunk_dir: Path) -> np.ndarray:
    dst_list = []
    files = list(raw_chunk_dir.glob("part_*/slot_*.pt"))
    for f in tqdm(files, desc="Scanning DSTs"):
        try:
            chunk = torch.load(f, map_location='cpu')
            dst_list.append(chunk['dst'].numpy())
            del chunk
        except: pass
    
    if not dst_list: return np.array([], dtype=np.int64)
    
    all_dst = np.concatenate(dst_list)
    pool = np.unique(all_dst) # 排序并去重
    print(f"[Pre-scan] Pool built. Size: {len(pool)}")
    return pool

def run_offline_materialization(
    raw_chunk_dir, output_root, num_nodes, layer_configs, 
    num_set = 8,num_neg=1, task_type='link', add_inverse=True,
    global_node_label=None, global_edge_label=None,
    node_parts=None, rep_table=None, pid=0, num_parts=1, partition_book=None
):
    print(f"\n[Step 6] Running Offline Materialization (Label Integrated)...")
    output_root.mkdir(parents=True, exist_ok=True)
    
    global_dst_pool_all = None
    if num_neg > 0:
        global_dst_pool_all = build_global_dst_pool(raw_chunk_dir)

    # [Fix] Only process directories
    part_dirs = sorted([d for d in raw_chunk_dir.glob("part_*") if d.is_dir()])
    
    for p_dir in part_dirs:
        curr_pid = int(p_dir.name.split('_')[1])
        pid = p_dir.name 
        print(f"  > Processing {pid}...")
        save_dir = output_root / pid
        save_dir.mkdir(parents=True, exist_ok=True)
        
        all_chunks = sorted(list(p_dir.glob("slot_*.pt")), key=lambda x: x.name)
        if len(all_chunks) == 0: continue
        print("    - Building Local ID Map...")
        p_nodes = torch.nonzero(node_parts == curr_pid, as_tuple=True)[0]
        # Use a tensor for fast lookup if num_nodes is reasonable (<100M)
        local_map = torch.full((num_nodes + 1,), -1, dtype=torch.long)
        local_map[p_nodes] = torch.arange(len(p_nodes), dtype=torch.long)
        hist_src, hist_dst, hist_ts, hist_eid, hist_cid = [], [], [], [], []
        if global_dst_pool_all is not None:
            global_dst_pool = global_dst_pool_all[torch.nonzero(node_parts[global_dst_pool_all] == curr_pid, as_tuple=True)[0]]
        print(f"    - Loading local history...")
        for c_path in tqdm(all_chunks, desc=f"Loading {pid}"):
            try:
                parts = c_path.stem.split('_')
                sub_id = int(parts[3])
                c_data = torch.load(c_path) 
                
                s, d, t, e = c_data['src'].numpy(), c_data['dst'].numpy(), c_data['ts'].numpy(), c_data['eid'].numpy()
                hist_src.append(s); hist_dst.append(d); hist_ts.append(t); hist_eid.append(e)
                hist_cid.append(np.full_like(s, sub_id))
                
                if add_inverse:
                    hist_src.append(d); hist_dst.append(s); hist_ts.append(t); hist_eid.append(e)
                    hist_cid.append(np.full_like(d, sub_id))
            except Exception as e:
                print(f"Error loading {c_path}: {e}")

        if not hist_src:
            print(f"    - Warning: No history loaded for {pid}")
            continue

        local_data = {
            'src': np.concatenate(hist_src), 'dst': np.concatenate(hist_dst),
            'ts': np.concatenate(hist_ts), 'eid': np.concatenate(hist_eid),
            'cid': np.concatenate(hist_cid)
        }
        del hist_src, hist_dst, hist_ts, hist_eid, hist_cid
        gc.collect()

        builder = MaterializedMultiLayerBuilder(
            local_data, num_nodes,
            node_parts=node_parts, 
            rep_table=rep_table, 
            pid=curr_pid, 
            num_parts=num_parts
            )
        builder.local_map = local_map
        print(f"    - Materializing batches...")
        for c_path in tqdm(all_chunks, desc=f"Sampling {pid}"):
            c_data = torch.load(c_path)
            parts = c_path.stem.split('_')
            sub_id = int(parts[3])
            
            chunk_route = c_data.get('route', None)
            
            if task_type == 'link':
                task_src = c_data['src'].numpy()
                task_dst = c_data['dst'].numpy()
                task_ts = c_data['ts'].numpy()
                task_eid = c_data['eid'].numpy()
                
                # [Fix] Assign Global Label (Already reordered in main)
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
                
                # [Fix] Assign Global Node Label (Already reordered in main)
                if global_node_label is not None:
                    if isinstance(global_node_label, torch.Tensor):
                        labels = global_node_label[node]
                        if isinstance(labels, torch.Tensor): labels = labels.numpy()
                    elif isinstance(global_node_label, list):
                        labels = np.zeros_like(node, dtype=np.int64)
                        t_min = node_ts.min()
                        t_max = node_ts.max()
                        for i in range(t_min, t_max + 1):
                            labels[node_ts == i] = global_node_label[i][node[node_ts == i]]
                        
                else:
                    labels = np.zeros_like(node, dtype=np.int64)
                    
                task_param = ('node', (node, node_ts, labels))
            
            builder.build_aligned_batch(
                task_param, save_dir / c_path.name, layer_configs, 
                num_neg, global_dst_pool, sub_id
            )
            if task_type == 'link' and num_set > 0:
                for i in range(num_set):
                    task_param = ('neg', task_ts)
                    builder.build_aligned_batch(
                        task_param, save_dir / f"neg_{i}_{c_path.name}", layer_configs, 
                        num_neg=1, global_dst_pool=global_dst_pool, cluster_ids=sub_id
                    )
        
        del local_data, builder
        gc.collect()

if __name__ == "__main__":
    src_root = Path("/mnt/data/zlj/starrygl-data/ctdg").resolve()
    tgt_root = Path("/mnt/data/zlj/starrygl-data/nparts").resolve()
    processed_root = Path("/mnt/data/zlj/starrygl-data/processed_atomic").resolve()
    
    num_inter_parts = 4
    hot_nodes_ratio = 0.1
    
    LAYER_CONFIGS = {
        'meta': {'start_offset': -1, 'end_offset': 0, 'graph_type': 'c'},
        'layer': [{'type': 'recent', 'fanout': 10}, {'type': 'recent', 'fanout': 10}]
    }

    for p in src_root.glob("*.pth"):
        name = p.stem
        if name in ['BCB']: continue
        if name != 'WIKI': continue
        print(f"Processing {name}...")
        
        raw_data_dict = torch.load(p, map_location='cpu')
        num_nodes = raw_data_dict['num_nodes']
        dataset = raw_data_dict['dataset']
        
        node_label = None
        edge_label = None
        perm,recv_perm = torch.load(tgt_root / f"{name}_{num_inter_parts:03d}"/"perm.pt")
        
        if isinstance(dataset, dict):
            if 'node_label' in dataset: 
                node_label = dataset['node_label'][perm]
            elif 'y' in dataset: 
                node_label = dataset['y'][perm]
            if 'edge_label' in dataset: edge_label = dataset['edge_label']
        elif isinstance(dataset, list):
            if 'node_label' in dataset[0]: 
                node_label = [d['node_label'][perm] for d in dataset]
            elif 'y' in dataset[0]: 
                node_label = [d['y'][perm] for d in dataset]
            
            if 'edge_label' in dataset[0]:
                edge_label = torch.cat([d['edge_label'] for d in dataset])  
        
        del raw_data_dict

        raw_dir = tgt_root / f"{name}_{num_inter_parts:03d}"
        if not raw_dir.exists(): 
            print(f"Skipping {name}: raw chunks not found")
            continue
        rep_table = torch.load(raw_dir / "replica_table.pt")
        partition_book_data = torch.load(raw_dir / "partition_book.pt")
        partition_book = partition_book_data[0]
        node_parts = partition_book_data[1]
        edge_parts = partition_book_data[2]
        print(node_parts, edge_parts)
        final_output_dir = processed_root / f"{name}_{num_inter_parts:03d}"
        
        run_offline_materialization(
            raw_dir, final_output_dir,
            num_nodes, LAYER_CONFIGS, num_neg=1, add_inverse=True,
            global_node_label=node_label,
            global_edge_label=edge_label,
            node_parts=node_parts,
            rep_table=rep_table, pid=0, num_parts=num_inter_parts,
            partition_book=partition_book
        )
         
        print(f"All done! Ready for training at {final_output_dir}")

# import sys
# import os
# from pathlib import Path
# import gc
# import torch
# import numpy as np
# from tqdm import tqdm
# import numba
# import multiprocessing as mp
# from functools import partial

# # [关键优化] 多进程模式下，必须限制 Numba 的线程数，否则 32进程 x 32线程 = 系统卡死
# os.environ["NUMBA_NUM_THREADS"] = "1" 
# os.environ["OMP_NUM_THREADS"] = "1"

# current_file = Path(__file__).resolve()
# project_root = current_file.parents[2] 
# sys.path.append(str(project_root))

# try:
#     from starrygl.cache.cache_route import CommPlan
# except ImportError:
#     pass

# # ==============================================================================
# # Part 1: Numba Kernel (无变化，注意环境变量已限制其并行度)
# # ==============================================================================
# @numba.njit(parallel=True, fastmath=True)
# def _jit_sample_kernel(
#     active_starts, active_ends, active_ts, active_indices, 
#     sorted_src, sorted_ts, sorted_eid, sorted_cluster,     
#     fanout, sample_type, graph_type,                       
#     start_offset, end_offset, cluster_id_val               
# ):
#     # 代码与之前相同，为节省篇幅略去...
#     # 核心逻辑不变，但在 NUMBA_NUM_THREADS=1 时会自动回退到单线程高效执行
#     # 由于我们在外层做了 Process 并行，这里单线程反而更快
#     num_tasks = len(active_starts)
#     counts = np.zeros(num_tasks, dtype=np.int64)
    
#     for i in numba.prange(num_tasks):
#         s = active_starts[i]
#         e = active_ends[i]
#         t_cut = active_ts[i]
#         nb_ts_view = sorted_ts[s:e]
#         cut_idx = np.searchsorted(nb_ts_view, t_cut + end_offset)
#         start_idx = np.searchsorted(nb_ts_view, t_cut + start_offset)
#         if cut_idx > len(nb_ts_view): cut_idx = len(nb_ts_view)
#         if start_idx < 0: start_idx = 0
#         if start_idx > cut_idx: start_idx = cut_idx
#         if cut_idx == 0:
#             counts[i] = 0
#             continue
#         count = 0
#         if graph_type == 0: 
#             if sample_type == 2: count = cut_idx 
#             elif sample_type == 0 or sample_type == 1: count = min(cut_idx, fanout)
#             else: count = cut_idx
#         else: 
#             if sample_type == 2:
#                 c_match = 0
#                 for k in range(s + start_idx, s + cut_idx):
#                     if sorted_cluster[k] == cluster_id_val: c_match += 1
#                 count = c_match
#             elif sample_type == 1:
#                 valid_len = cut_idx - start_idx
#                 count = min(valid_len, fanout) if valid_len > 0 else 0
#             else:
#                 count = max(0, cut_idx - start_idx)
#         counts[i] = count

#     total_edges = np.sum(counts)
#     offsets = np.zeros(num_tasks + 1, dtype=np.int64)
#     offsets[1:] = np.cumsum(counts)
    
#     out_src = np.empty(total_edges, dtype=sorted_src.dtype)
#     out_dst = np.empty(total_edges, dtype=active_indices.dtype)
#     out_ts  = np.empty(total_edges, dtype=sorted_ts.dtype)
#     out_dt  = np.empty(total_edges, dtype=np.float32) 
#     out_eid = np.empty(total_edges, dtype=sorted_eid.dtype)
    
#     for i in numba.prange(num_tasks):
#         count = counts[i]
#         if count == 0: continue
#         write_start = offsets[i]
#         target_id = active_indices[i]
#         t_cut = active_ts[i]
#         s, e = active_starts[i], active_ends[i]
#         nb_ts_view = sorted_ts[s:e]
#         cut_idx = np.searchsorted(nb_ts_view, t_cut + end_offset)
#         start_idx = np.searchsorted(nb_ts_view, t_cut + start_offset)
#         if cut_idx > len(nb_ts_view): cut_idx = len(nb_ts_view)
#         if start_idx < 0: start_idx = 0
#         if start_idx > cut_idx: start_idx = cut_idx
#         current_ptr = write_start
        
#         if graph_type == 0: # continuous
#             real_start, real_end = s, s
#             if sample_type == 2: real_start, real_end = s, s + cut_idx
#             elif sample_type == 0:
#                 offset_back = max(0, cut_idx - fanout)
#                 real_start, real_end = s + offset_back, s + cut_idx
#             elif sample_type == 1:
#                 if cut_idx <= fanout: real_start, real_end = s, s + cut_idx
#                 else:
#                     rand_idxs = np.random.choice(cut_idx, fanout, replace=False)
#                     rand_idxs.sort() 
#                     for k in range(fanout):
#                         idx = s + rand_idxs[k]
#                         out_src[current_ptr] = sorted_src[idx]
#                         out_ts[current_ptr]  = sorted_ts[idx]
#                         out_eid[current_ptr] = sorted_eid[idx]
#                         out_dst[current_ptr] = target_id
#                         out_dt[current_ptr]  = t_cut - sorted_ts[idx]
#                         current_ptr += 1
#                     continue 
#             else: real_start, real_end = s, s + cut_idx
            
#             if current_ptr == write_start: 
#                 w_len = real_end - real_start
#                 if w_len > 0:
#                     out_src[write_start : write_start + w_len] = sorted_src[real_start : real_end]
#                     out_ts[write_start : write_start + w_len]  = sorted_ts[real_start : real_end]
#                     out_eid[write_start : write_start + w_len] = sorted_eid[real_start : real_end]
#                     out_dst[write_start : write_start + w_len] = target_id 
#                     out_dt[write_start : write_start + w_len]  = t_cut - sorted_ts[real_start : real_end]
#         else:
#              real_start = s + start_idx
#              real_end = s + cut_idx
#              w_len = real_end - real_start
#              if w_len > 0 and count > 0:
#                  copy_len = min(w_len, count)
#                  out_src[write_start:write_start+copy_len] = sorted_src[real_start:real_start+copy_len]
#                  out_ts[write_start:write_start+copy_len] = t_cut
#                  out_eid[write_start:write_start+copy_len] = sorted_eid[real_start:real_start+copy_len]
#                  out_dst[write_start:write_start+copy_len] = target_id
#                  out_dt[write_start:write_start+copy_len] = t_cut - sorted_ts[real_start:real_start+copy_len]
#     return out_src, out_dst, out_ts, out_dt, out_eid

# # ==============================================================================
# # Part 2: Materialized Builder
# # ==============================================================================
    
# class MaterializedMultiLayerBuilder:
#     def __init__(self, partition_data, num_nodes, node_parts, rep_table, pid, num_parts):
#         self.hist_src = partition_data['src']
#         self.hist_dst = partition_data['dst']
#         self.hist_ts = partition_data['ts']
#         self.hist_cluster = partition_data.get('cid', np.zeros_like(self.hist_src, dtype=np.int64))
#         self.hist_eid = partition_data.get('eid', np.arange(len(self.hist_src), dtype=np.int64))
#         self.num_nodes = num_nodes
#         self.node_parts = node_parts 
#         self.rep_table = rep_table
#         self.pid = pid
#         self.num_parts = num_parts
        
#         # 预排序
#         sort_keys = np.lexsort((self.hist_ts, self.hist_dst))
#         self.sorted_src = self.hist_src[sort_keys]
#         self.sorted_dst = self.hist_dst[sort_keys]
#         self.sorted_ts = self.hist_ts[sort_keys]
#         self.sorted_eid = self.hist_eid[sort_keys]
#         self.sorted_cluster = self.hist_cluster[sort_keys]
        
#         # 索引构建
#         self.full_indptr = np.zeros(num_nodes + 1, dtype=np.int64)
#         u, c = np.unique(self.sorted_dst, return_counts=True)
#         valid_mask = u < num_nodes
#         self.full_indptr[u[valid_mask]+1] = c[valid_mask]
#         np.cumsum(self.full_indptr, out=self.full_indptr)

#     def _get_unique_pairs(self, nodes, timestamps):
#         dt = np.dtype([('id', nodes.dtype), ('ts', timestamps.dtype)])
#         combined = np.empty(len(nodes), dtype=dt)
#         combined['id'] = nodes
#         combined['ts'] = timestamps
#         u_combined, inverse = np.unique(combined, return_inverse=True)
#         return u_combined['id'], u_combined['ts'], inverse

#     def _build_csc_from_pairs(self, src_local, dst_local, ts, dt, eid, num_dst_nodes):
#         if len(src_local) == 0:
#             return (np.zeros(num_dst_nodes + 1, dtype=np.int64), np.array([], dtype=np.int64), 
#                     np.array([], dtype=np.int64), np.array([], dtype=np.float32), np.array([], dtype=np.int64))
#         perm = np.lexsort((ts, dst_local))
#         s_src, s_dst, s_ts, s_dt, s_eid = src_local[perm], dst_local[perm], ts[perm], dt[perm], eid[perm]
        
#         indptr = np.zeros(num_dst_nodes + 1, dtype=np.int64)
#         u, c = np.unique(s_dst, return_counts=True)
#         valid_mask = u < num_dst_nodes
#         indptr[u[valid_mask]+1] = c[valid_mask]
#         np.cumsum(indptr, out=indptr)
#         return indptr, s_src, s_ts, s_dt, s_eid

#     def _merge_and_map_nodes(self, curr_ids, curr_ts, raw_ids, raw_ts):
#         dt = np.dtype([('id', curr_ids.dtype), ('ts', curr_ts.dtype)])
#         all_struct = np.empty(len(curr_ids) + len(raw_ids), dtype=dt)
#         all_struct['id'][:len(curr_ids)] = curr_ids
#         all_struct['ts'][:len(curr_ids)] = curr_ts
#         all_struct['id'][len(curr_ids):] = raw_ids
#         all_struct['ts'][len(curr_ids):] = raw_ts
#         unique_struct = np.unique(all_struct)
#         raw_struct_view = all_struct[len(curr_ids):]
#         final_inverse = np.searchsorted(unique_struct, raw_struct_view)
#         return unique_struct['id'], unique_struct['ts'], final_inverse
    
#     def _sample_layer(self, target_nodes, target_ts, fanout=10, sample_type='recent', meta_config=None, cluster_ids=None):
#         starts = self.full_indptr[target_nodes]
#         ends = self.full_indptr[target_nodes + 1]
#         valid_mask = (ends - starts) > 0
#         active_idx = np.where(valid_mask)[0]
#         if len(active_idx) == 0: return None, None, None, None, None
        
#         s_type = {'recent':0, 'uniform':1, 'cluster':2, 'full':3}.get(sample_type, 0)
#         g_type = 0 if meta_config.get('graph_type', 'c') == 'c' else 1
#         cid = cluster_ids if cluster_ids is not None else -1
        
#         return _jit_sample_kernel(
#             starts[valid_mask], ends[valid_mask], target_ts[valid_mask], active_idx,
#             self.sorted_src, self.sorted_ts, self.sorted_eid, self.sorted_cluster,
#             int(fanout), int(s_type), int(g_type),
#             int(meta_config.get('start_offset', 0)), int(meta_config.get('end_offset', 0)), int(cid)
#         )

#     def _compute_layer_route(self, layer_nodes):
#         if len(layer_nodes) == 0: return None
#         u_nodes = torch.from_numpy(layer_nodes) if isinstance(layer_nodes, np.ndarray) else layer_nodes
#         owners = self.node_parts[u_nodes]
#         my_nodes = u_nodes[owners == self.pid]
#         if len(my_nodes) == 0: return None
#         q_idx, q_ranks, q_remote_locs = self.rep_table.lookup(my_nodes)
#         if len(q_ranks) == 0: return None
        
#         s_gids = my_nodes[q_idx]
#         s_local = s_gids 
#         sort_idx = torch.argsort(q_ranks)
#         final_ranks = q_ranks[sort_idx]
#         final_local = s_local[sort_idx]
#         final_remote = q_remote_locs[sort_idx]
        
#         u_ranks, counts = torch.unique(final_ranks, return_counts=True)
#         send_sizes = torch.zeros(self.num_parts, dtype=torch.long)
#         send_sizes[u_ranks.long()] = counts.long()
#         return CommPlan(send_ranks=final_ranks, send_sizes=send_sizes, send_indices=final_local, send_remote_indices=final_remote)


#     def build_aligned_batch(self, task_param, layer_configs, num_neg=1, global_dst_pool=None, cluster_ids=None):
#         task_type, task_nodes = task_param
#         if task_type == 'link':
#             task_src, task_dst, task_ts, task_label, task_eid = task_nodes 
#             l0_nodes = np.concatenate([task_src, task_dst])
#             l0_ts = np.concatenate([task_ts, task_ts])
#         elif task_type == 'node':
#             task_node, task_ts, task_label = task_nodes
#             l0_nodes, l0_ts = task_node, task_ts
#             neg_dst_pool = None
#         elif task_type == 'neg':
#             if num_neg > 0 and global_dst_pool is not None:
#                 task_ts = task_nodes
#                 rand_indices = np.random.randint(0, len(global_dst_pool), (len(task_ts), num_neg)).reshape(-1)
#                 neg_dst_pool = global_dst_pool[rand_indices]
#                 l0_nodes = neg_dst_pool.flatten()
#                 l0_ts = np.repeat(task_ts, num_neg)
#             else: return None

#         curr_uid, curr_uts, l0_inv = self._get_unique_pairs(l0_nodes, l0_ts)
#         batch_data = [{"gids": curr_uid, "ts": curr_uts, "inv_map": l0_inv, "num_neg": num_neg}]
#         layer_routes = []
#         r0 = self._compute_layer_route(curr_uid)
#         layer_routes.append(r0)
        
#         last_inv_map = None
#         for l, config in enumerate(layer_configs['layer']):
#             target_u = curr_uid if l==0 else curr_uid[last_inv_map]
#             target_t = curr_uts if l==0 else curr_uts[last_inv_map]
            
#             raw_src, raw_dst, raw_ts, raw_dt, raw_eid = self._sample_layer(
#                 target_u, target_t, config['fanout'], config['type'], layer_configs['meta'], cluster_ids
#             )
            
#             if raw_src is None:
#                 curr_uid = np.array([], dtype=curr_uid.dtype)
#                 curr_uts = np.array([], dtype=curr_uts.dtype)
#                 batch_data.append({
#                     "indptr": np.zeros(1, dtype=np.int64), "indices": np.array([], dtype=np.int64),
#                     "eid": np.array([], dtype=np.int64), "edge_ts": np.array([], dtype=curr_uts.dtype),
#                     "edge_dt": np.array([], dtype=np.float32), "gids": curr_uid, "ts": curr_uts
#                 })
#                 layer_routes.append(None)
#                 break
                
#             raw_dst_mapped = last_inv_map[raw_dst] if last_inv_map is not None else raw_dst
#             next_uid, next_uts, next_inv = self._merge_and_map_nodes(curr_uid, curr_uts, raw_src, raw_ts)
            
#             ptr, idx, b_ts, b_dt, b_eid = self._build_csc_from_pairs(
#                 next_inv, raw_dst_mapped, raw_ts, raw_dt, raw_eid, curr_uid.shape[0]
#             )
   
#             batch_data.append({
#                 "indptr": ptr, "indices": idx, "eid": b_eid, "edge_ts": b_ts,
#                 "edge_dt": b_dt, "gids": next_uid, "ts": next_uts
#             })
#             if l < len(layer_configs['layer']) - 1:
#                 r_next = self._compute_layer_route(next_uid)
#                 layer_routes.append(r_next)
#             curr_uid, curr_uts, last_inv_map = next_uid, next_uts, next_inv
        
#         def to_tensor(x): return torch.from_numpy(x) if isinstance(x, np.ndarray) else x
#         final_data = []
#         for d in batch_data:
#             final_data.append({k: to_tensor(v) for k,v in d.items()})
            
#         if task_type == 'link':
#             final_data[0].update({
#                 "task_ts": to_tensor(task_ts), "task_src": to_tensor(task_src),
#                 "task_dst": to_tensor(task_dst), 
#                 "task_label": to_tensor(task_label), "task_eid": to_tensor(task_eid) 
#             })
#         elif task_type == 'node':
#             final_data[0].update({
#                 "task_node": to_tensor(task_node), "task_ts": to_tensor(task_ts), 
#                 "task_label": to_tensor(task_label) 
#             })
#         elif task_type == 'neg':
#             final_data[0].update({
#                 "task_nodes": to_tensor(neg_dst_pool), "task_ts": to_tensor(task_ts),
#                 'task_label': 'neg_set'
#             })
#         final_data[0]['comm_plan'] = layer_routes
#         return final_data

# # ==============================================================================
# # Part 3: Worker Functions for Multiprocessing
# # ==============================================================================

# # 全局变量，用于 worker 进程中共享数据（利用 Linux fork 的 Copy-On-Write 机制，无需 Pickling）
# WORKER_BUILDER = None
# WORKER_DST_POOL = None
# WORKER_EDGE_LABEL = None

# def init_worker(builder, dst_pool, edge_label):
#     """
#     初始化 Worker 进程。
#     注意：在 Linux 上，如果使用 fork 启动，这一步其实是冗余的，
#     因为子进程会自动继承父进程的内存空间。
#     但为了兼容性和规范，我们在这里显式声明全局引用。
#     """
#     global WORKER_BUILDER, WORKER_DST_POOL, WORKER_EDGE_LABEL
#     WORKER_BUILDER = builder
#     WORKER_DST_POOL = dst_pool
#     WORKER_EDGE_LABEL = edge_label

# def process_chunk_task(task_payload):
#     """
#     这是每个 Worker 进程实际执行的函数。
#     task_payload: (batch_id, list_of_file_paths, task_type, layer_configs, num_neg, num_set, sub_id)
#     """
#     batch_id, chunk_group, task_type, layer_configs, num_neg, num_set, sub_id, save_dir = task_payload
    
#     # 1. Load Data
#     group_src, group_dst, group_ts, group_eid = [], [], [], []
#     for c_path in chunk_group:
#         c_data = torch.load(c_path)
#         group_src.append(c_data['src'].numpy())
#         group_dst.append(c_data['dst'].numpy())
#         group_ts.append(c_data['ts'].numpy())
#         group_eid.append(c_data['eid'].numpy())
    
#     mega_src = np.concatenate(group_src)
#     mega_dst = np.concatenate(group_dst)
#     mega_ts  = np.concatenate(group_ts)
#     mega_eid = np.concatenate(group_eid)
    
#     # 2. Get Label
#     if task_type == 'link':
#         if WORKER_EDGE_LABEL is not None:
#             labels = WORKER_EDGE_LABEL[mega_eid]
#             if isinstance(labels, torch.Tensor): labels = labels.numpy()
#         else:
#             labels = np.ones_like(mega_src, dtype=np.int64)
#         task_param = ('link', (mega_src, mega_dst, mega_ts, labels, mega_eid))
#     else:
#         # Placeholder for node tasks if needed
#         return 

#     # 3. Positive Sample
#     pos_result = WORKER_BUILDER.build_aligned_batch(
#         task_param, layer_configs, num_neg, WORKER_DST_POOL, sub_id
#     )
    
#     # 4. Save Positive
#     # start_idx = chunk_group[0] name ... just use batch_id
#     save_name = f"packed_{batch_id:06d}.pt"
#     torch.save(pos_result, save_dir / save_name)
    
#     # 5. Negative Sample (Loop)
#     if task_type == 'link' and num_set > 0:
#         for n_i in range(num_set):
#             neg_param = ('neg', mega_ts)
#             neg_result = WORKER_BUILDER.build_aligned_batch(
#                 neg_param, layer_configs, num_neg=1, 
#                 global_dst_pool=WORKER_DST_POOL, cluster_ids=sub_id
#             )
#             neg_save_name = f"packed_neg_{n_i}_{batch_id:06d}.pt"
#             torch.save(neg_result, save_dir / neg_save_name)
            
#     return batch_id # Return something just to signal done

# def _load_dst_file(f_path):
#     # 辅助函数：快速加载 DST
#     try:
#         c = torch.load(f_path, map_location='cpu')
#         return c['dst'].numpy()
#     except:
#         return np.array([], dtype=np.int64)

# # ==============================================================================
# # Part 4: Main Execution
# # ==============================================================================

# def build_global_dst_pool_parallel(raw_chunk_dir: Path) -> np.ndarray:
#     files = list(raw_chunk_dir.glob("part_*/slot_*.pt"))
#     print(f"    - Scanning {len(files)} files for DST pool using Multiprocessing...")
    
#     dst_list = []
#     # 使用 ProcessPoolExecutor 进行并行文件读取
#     # 注意：Pool的大小不要超过CPU核心数，这里保守取 16
#     with mp.Pool(processes=16) as pool:
#         for res in tqdm(pool.imap_unordered(_load_dst_file, files, chunksize=100), total=len(files), desc="Parallel Scan"):
#             if len(res) > 0:
#                 dst_list.append(res)
                
#     if not dst_list: return np.array([], dtype=np.int64)
#     all_dst = np.concatenate(dst_list)
#     pool = np.unique(all_dst) 
#     print(f"    - Pool built. Size: {len(pool)}")
#     return pool

# def run_batched_materialization(
#     raw_chunk_dir, output_root, num_nodes, layer_configs, 
#     num_set=8, num_neg=1, task_type='link', add_inverse=True,
#     global_node_label=None, global_edge_label=None,
#     node_parts=None, rep_table=None, pid=0, num_parts=1, partition_book=None,
#     BATCH_GROUP_SIZE = 100 
# ):
#     print(f"\n[Step 6] Running Parallel Materialization...")
#     output_root.mkdir(parents=True, exist_ok=True)
    
#     global_dst_pool_all = None
#     if num_neg > 0:
#         global_dst_pool_all = build_global_dst_pool_parallel(raw_chunk_dir)

#     part_dirs = sorted([d for d in raw_chunk_dir.glob("part_*") if d.is_dir()])
    
#     for p_dir in part_dirs:
#         curr_pid = int(p_dir.name.split('_')[1])
#         pid_name = p_dir.name 
#         print(f"  > Processing Partition {pid_name}...")
#         save_dir = output_root / pid_name
#         save_dir.mkdir(parents=True, exist_ok=True)
        
#         # 1. Load History (单进程加载，这部分很快且必须完整)
#         all_chunks = sorted(list(p_dir.glob("slot_*.pt")), key=lambda x: x.name)
#         if len(all_chunks) == 0: continue
        
#         print("    - Loading Local History (Single Process)...")
#         hist_src, hist_dst, hist_ts, hist_eid, hist_cid = [], [], [], [], []
        
#         # 这里的加载可以并行，但为了简单起见保持串行，因为它只占总时间的 <5%
#         for c_path in tqdm(all_chunks, desc=f"Loading History"):
#             try:
#                 parts = c_path.stem.split('_')
#                 sub_id = int(parts[3])
#                 c_data = torch.load(c_path) 
#                 s, d, t, e = c_data['src'].numpy(), c_data['dst'].numpy(), c_data['ts'].numpy(), c_data['eid'].numpy()
#                 hist_src.append(s); hist_dst.append(d); hist_ts.append(t); hist_eid.append(e)
#                 hist_cid.append(np.full_like(s, sub_id))
#                 if add_inverse:
#                     hist_src.append(d); hist_dst.append(s); hist_ts.append(t); hist_eid.append(e)
#                     hist_cid.append(np.full_like(d, sub_id))
#             except: pass

#         local_data = {
#             'src': np.concatenate(hist_src), 'dst': np.concatenate(hist_dst),
#             'ts': np.concatenate(hist_ts), 'eid': np.concatenate(hist_eid),
#             'cid': np.concatenate(hist_cid)
#         }
#         del hist_src, hist_dst, hist_ts, hist_eid, hist_cid
#         gc.collect()

#         # 2. Init Builder (单进程构建，包含排序)
#         print("    - Initializing Builder and Sorting Indices...")
#         builder = MaterializedMultiLayerBuilder(
#             local_data, num_nodes, node_parts=node_parts, 
#             rep_table=rep_table, pid=curr_pid, num_parts=num_parts
#         )
        
#         # 3. Prepare Parallel Tasks
#         tasks = []
#         total_files = len(all_chunks)
#         # cluster id logic (simplified)
#         current_sub_id = int(all_chunks[0].stem.split('_')[3]) if all_chunks else 0
        
#         for i in range(0, total_files, BATCH_GROUP_SIZE):
#             chunk_group = all_chunks[i : min(i + BATCH_GROUP_SIZE, total_files)]
#             if not chunk_group: continue
            
#             # Payload: (batch_id, files, type, config, n_neg, n_set, sub_id, save_dir)
#             tasks.append((
#                 i, chunk_group, task_type, layer_configs, num_neg, num_set, current_sub_id, save_dir
#             ))
            
#         print(f"    - Spawning {len(tasks)} Mega-Batch Tasks across {mp.cpu_count()} cores...")
        
#         # 4. Run Parallel Pool
#         # 这里使用 fork，子进程会“免费”获得 builder 的内存视图
#         # 设置 chunksize=1 可以让进度条更平滑
#         # num_workers 建议设置为 CPU 核心数的 80%
#         num_workers = max(1, mp.cpu_count() - 2) 
        
#         with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(builder, global_dst_pool_all, global_edge_label)) as pool:
#             for _ in tqdm(pool.imap_unordered(process_chunk_task, tasks, chunksize=1), total=len(tasks), desc="Parallel Processing"):
#                 pass
                
#         # Cleanup for next partition
#         del local_data, builder
#         gc.collect()

# if __name__ == "__main__":
#     # 配置区
#     src_root = Path("/mnt/data/zlj/starrygl-data/ctdg").resolve()
#     tgt_root = Path("/mnt/data/zlj/starrygl-data/nparts").resolve()
#     processed_root = Path("/mnt/data/zlj/starrygl-data/processed_atomic").resolve()
    
#     num_inter_parts = 4
#     # GDELT 推荐 100
#     BATCH_GROUP_SIZE = 100 
    
#     LAYER_CONFIGS = {
#         'meta': {'start_offset': -1, 'end_offset': 0, 'graph_type': 'c'},
#         'layer': [{'type': 'recent', 'fanout': 10}, {'type': 'recent', 'fanout': 10}]
#     }

#     # 确保 multiprocessing 启动方式安全 (在某些系统上 fork 是默认的，但在 PyTorch 环境中有时需要注意)
#     try:
#         mp.set_start_method('fork')
#     except RuntimeError:
#         pass #Context already set

#     for p in src_root.glob("*.pth"):
#         name = p.stem
#         if name in ['BCB']: continue
#         if name != 'WIKI': continue
#         # if name != 'GDELT': continue 
#         print(f"Processing {name}...")
        
#         raw_data_dict = torch.load(p, map_location='cpu')
#         num_nodes = raw_data_dict['num_nodes']
#         dataset = raw_data_dict['dataset']
#         edge_label = None
#         perm_path = tgt_root / f"{name}_{num_inter_parts:03d}"/"perm.pt"
#         if perm_path.exists():
#             perm, recv_perm = torch.load(perm_path)
#             if isinstance(dataset, dict):
#                 if 'edge_label' in dataset: edge_label = dataset['edge_label']
#             elif isinstance(dataset, list):
#                 if 'edge_label' in dataset[0]:
#                     edge_label = torch.cat([d['edge_label'] for d in dataset])
#         del raw_data_dict

#         raw_dir = tgt_root / f"{name}_{num_inter_parts:03d}"
#         if not raw_dir.exists(): continue
            
#         rep_table = torch.load(raw_dir / "replica_table.pt")
#         partition_book_data = torch.load(raw_dir / "partition_book.pt")
#         partition_book = partition_book_data[0]
#         node_parts = partition_book_data[1]
        
#         final_output_dir = processed_root / f"{name}_{num_inter_parts:03d}"
        
#         run_batched_materialization(
#             raw_dir, final_output_dir,
#             num_nodes, LAYER_CONFIGS, num_neg=1, add_inverse=True,
#             global_node_label=None, # pass None to save memory if not needed in func
#              global_edge_label=edge_label,
#             node_parts=node_parts,
#             rep_table=rep_table, pid=0, num_parts=num_inter_parts,
#             partition_book=partition_book,
#             BATCH_GROUP_SIZE=BATCH_GROUP_SIZE
#         )
         
#         print(f"Done! Saved to {final_output_dir}")