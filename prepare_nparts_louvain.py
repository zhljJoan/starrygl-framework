import os
import gc
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
import dgl
import networkx as nx
import pymetis
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from tqdm import tqdm
from joblib import Parallel, delayed
import community as community_louvain 

# ==============================================================================
# Mock Imports
# ==============================================================================
try:
    from starrygl.cache.cache_route import CommPlan
    from starrygl.cache.replica_table import build_replica_table
except ImportError:
    class CommPlan:
        def __init__(self, send_ranks, send_sizes, send_indices, send_remote_indices): pass
    def build_replica_table(num_nodes, partition_book, num_parts): return None

# ==============================================================================
# Part 1: Partitioning (优化图构建)
# ==============================================================================

def partition_hybrid_manifold(num_nodes, edge_index, edge_ts, hot_mask, num_parts, part_type='louvain', num_micro_parts=128):
    print("[Step 1] Running Hybrid Manifold Partitioning...")
    # 尽可能保持在 GPU 做 mask 计算，最后转 numpy
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    ts = edge_ts.cpu().numpy()

    if part_type == 'louvain':
        if len(ts) > 0:
            # [优化] 避免 zip(src, dst) 的 Python 循环，虽然 nx 依然慢，但构建列表是瓶颈之一
            # 如果可能，建议这里使用 pymetis 或者 cugraph (如果可用)
            nx_graph = nx.Graph()
            nx_graph.add_nodes_from(np.arange(num_nodes))
            
            # 批量添加边 (NetworkX 优化写法)
            edges = np.stack([src, dst], axis=1)
            nx_graph.add_edges_from(edges)
            
            print(f"    - Micro-clustering via Louvain ({nx_graph.number_of_nodes()} nodes)...")
            partition_map = community_louvain.best_partition(nx_graph, resolution=3.0)
        else:
            partition_map = {}
        
        node_to_cluster = np.full(num_nodes, -1, dtype=np.int32)
        if partition_map:
            nodes = np.array(list(partition_map.keys()))
            clusters = np.array(list(partition_map.values()))
            node_to_cluster[nodes] = clusters
            
    elif part_type == 'metis':
        print(f"    - Micro-clustering via Metis ({num_micro_parts} parts)...")
        # Pymetis 需要 adj list，这是 Python 循环瓶颈
        # 使用 scipy csr 转 adj list 会快很多
        adj_csr = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes)).tocsr()
        # 对称化
        adj_csr = adj_csr + adj_csr.T
        adj_list = [adj_csr.indices[adj_csr.indptr[i]:adj_csr.indptr[i+1]] for i in range(num_nodes)]
        
        _, membership = pymetis.part_graph(num_micro_parts, adjacency=adj_list)
        node_to_cluster = np.array(membership, dtype=np.int32)
    
    # ... (后续逻辑保持不变，这部分 Numpy 操作很快) ...
    max_cluster_id = node_to_cluster.max() + 1
    unclustered_mask = node_to_cluster == -1
    isolated_nodes = np.where(unclustered_mask)[0]
    if len(isolated_nodes) > 0:
        new_ids = np.arange(max_cluster_id, max_cluster_id + len(isolated_nodes))
        node_to_cluster[isolated_nodes] = new_ids
        max_cluster_id += len(isolated_nodes)
    
    print("    - Analyzing cluster temporal properties...")
    edge_clusters = node_to_cluster[dst] 
    
    if max_cluster_id < 2000000: # 内存安全阈值
        count = np.bincount(edge_clusters, minlength=max_cluster_id)
        time_sum = np.bincount(edge_clusters, weights=ts, minlength=max_cluster_id)
        cluster_mean_ts = np.zeros(max_cluster_id)
        valid_c = count > 0
        cluster_mean_ts[valid_c] = time_sum[valid_c] / count[valid_c]
    else:
        cluster_mean_ts = np.zeros(max_cluster_id) # Fallback

    unique_cids = np.where(np.bincount(node_to_cluster) > 0)[0]
    sorted_cids = unique_cids[np.argsort(cluster_mean_ts[unique_cids])]
    
    cluster_to_part = np.zeros(max_cluster_id, dtype=np.int32)
    chunk_size = (len(sorted_cids) + num_parts - 1) // num_parts
    
    for i, cid in enumerate(sorted_cids):
        pid = i // chunk_size
        if pid >= num_parts: pid = num_parts - 1
        cluster_to_part[cid] = pid
        
    final_parts = torch.from_numpy(cluster_to_part[node_to_cluster]).long()
    print(f"    - Final partition stats: {final_parts.unique(return_counts=True)}")
    return final_parts, node_to_cluster

# ==============================================================================
# Part 2: Reordering (并行化优化)
# ==============================================================================

def _process_single_partition_rcm(pid, p_nodes, full_adj_indices, full_adj_indptr, hot_mask_np, node_avg_ts_np, num_time_buckets):
    """
    Worker function for parallel RCM. 
    传递 CSR 的 raw array 以避免 pickle 整个 graph 对象。
    """
    if len(p_nodes) == 0: return np.array([], dtype=np.int64)
    
    is_hot = hot_mask_np[p_nodes]
    local_hubs = p_nodes[is_hot]
    local_cold = p_nodes[~is_hot]
    
    # 1. Hubs: 简单按原始顺序或重新度数计算（这里简化，不重新算度数以加速）
    # 如果必须按度数，需要传度数数组进来
    
    # 2. Cold: RCM
    cold_layout = []
    if len(local_cold) > 0:
        times = node_avg_ts_np[local_cold]
        sorted_cold = local_cold[np.argsort(times)]
        buckets = np.array_split(sorted_cold, num_time_buckets)
        
        for bucket in buckets:
            if len(bucket) == 0: continue
            
            # 手动提取子图 CSR (比 scipy indexing 快)
            # 这里为了代码简洁，还是用 scipy 的切片，但因为是在子进程且数据量小，是可以接受的
            # 重建 CSR
            data = np.ones(len(full_adj_indices), dtype=np.int8) # Dummy data
            mat = sp.csr_matrix((data, full_adj_indices, full_adj_indptr))
            sub_csr = mat[bucket, :][:, bucket]
            
            perm = reverse_cuthill_mckee(sub_csr, symmetric_mode=False)
            cold_layout.append(bucket[perm])
            
    local_cold_sorted = np.concatenate(cold_layout) if cold_layout else np.array([], dtype=np.int64)
    return np.concatenate([local_hubs, local_cold_sorted])

def hierarchical_spatiotemporal_reordering(graph, node_parts, hot_mask, node_avg_ts, num_parts, num_time_buckets=8):
    print(f"\n[Step 2] Calculating Reordering Permutation (Parallel)...")
    
    node_parts_np = node_parts.cpu().numpy()
    hot_mask_np = hot_mask.cpu().numpy()
    node_avg_ts_np = node_avg_ts.cpu().numpy()
    
    # 获取 CSR 数组
    try:
        adj = graph.adj_external(scipy_fmt='csr')
    except:
        adj = graph.adj(scipy_fmt='csr')
    
    # 将 CSR 组件放入共享内存 (Joblib 在 fork 模式下效率较高)
    indices = adj.indices
    indptr = adj.indptr
    
    # 并行计算 

    #[Image of Parallel Processing]

    results = Parallel(n_jobs=min(num_parts, 16), backend="loky")(
        delayed(_process_single_partition_rcm)(
            pid, 
            np.where(node_parts_np == pid)[0],
            indices, indptr, 
            hot_mask_np, node_avg_ts_np, 
            num_time_buckets
        ) for pid in range(num_parts)
    )
    
    full_perm = np.concatenate(results)
    
    # 补全
    if len(full_perm) != graph.num_nodes():
        mask = np.ones(graph.num_nodes(), dtype=bool)
        mask[full_perm] = False
        full_perm = np.concatenate([full_perm, np.where(mask)[0]])
            
    return torch.from_numpy(full_perm)

# ==============================================================================
# Part 3: Chunk Generation (I/O 并行优化)
# ==============================================================================

def _save_chunk_task(chunk_data, save_path):
    """IO Worker"""
    torch.save(chunk_data, save_path)

def prepare_spatiotemporal_chunks(
    edge_index: torch.Tensor, edge_ts: torch.Tensor, 
    node_parts: torch.Tensor, node_clusters: np.ndarray,
    num_parts: int, slice_param: Tuple[str, int], 
    output_dir: Path
):
    print(f"\n[Step 3] Generating Spatio-Temporal Chunks (Async IO)...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 时间边界 (Vectorized)
    if slice_param[0] == "event":
        u_ts, counts = torch.unique(edge_ts, return_counts=True)
        cum = torch.cumsum(counts, 0)
        # ... logic consistent with original ...
        boundaries = [edge_ts.min().item()]
        curr = 0
        total = cum[-1].item()
        cum_cpu = cum.cpu().numpy()
        u_ts_cpu = u_ts.cpu().numpy()
        while curr < total:
            target = curr + slice_param[1]
            if target >= total: break
            idx = np.searchsorted(cum_cpu, target)
            boundaries.append(u_ts_cpu[idx])
            curr = cum_cpu[idx]
        if boundaries[-1] < edge_ts.max().item(): boundaries.append(edge_ts.max().item() + 1)
        time_boundaries = torch.tensor(boundaries).to(edge_ts.device) # Keep on GPU
    else:
        time_boundaries = torch.linspace(edge_ts.min(), edge_ts.max() + 1, slice_param[1]).to(edge_ts.device)
        
    torch.save({"boundaries": time_boundaries.cpu(), "strategy": slice_param}, output_dir / "dist_meta.pt")
    
    src, dst = edge_index
    edge_owners = node_parts[dst]
    node_clusters_t = torch.from_numpy(node_clusters).to(edge_index.device).long()
    eids = torch.arange(len(src), device=src.device)
    
    # [优化] 使用 ThreadPoolExecutor 进行异步写盘 
    io_pool = ThreadPoolExecutor(max_workers=16) 
    futures = []

    for pid in tqdm(range(num_parts), desc="Partitions"):
        part_dir = output_dir / f"part_{pid}"
        part_dir.mkdir(exist_ok=True)
        
        # GPU 上筛选
        mask = (edge_owners == pid)
        p_src, p_dst, p_ts, p_eid = src[mask], dst[mask], edge_ts[mask], eids[mask]
        
        # GPU 上 Bucketize
        # time_boundaries 必须在 GPU 上
        edge_slots = torch.bucketize(p_ts, time_boundaries, right=True) - 1
        valid_mask = (edge_slots >= 0) & (edge_slots < len(time_boundaries) - 1)
        
        p_src = p_src[valid_mask]
        p_dst = p_dst[valid_mask]
        p_ts = p_ts[valid_mask]
        p_eid = p_eid[valid_mask]
        edge_slots = edge_slots[valid_mask]
        
        p_cids = node_clusters_t[p_dst]
        
        # 联合排序: (Slot, Cluster, Time) -> 转 Numpy lexsort (PyTorch 尚无稳定 lexsort)
        # 为了速度，我们将排序键转到 CPU，数据可以留在 GPU 等切分完再转
        keys_cpu = [
            p_ts.cpu().numpy(), 
            p_cids.cpu().numpy(), 
            edge_slots.cpu().numpy()
        ]
        sort_idx = np.lexsort(keys_cpu) # Key order: last is primary
        sort_idx_t = torch.from_numpy(sort_idx).to(src.device)
        
        # Reorder on GPU
        s_src = p_src[sort_idx_t]
        s_dst = p_dst[sort_idx_t]
        s_ts  = p_ts[sort_idx_t]
        s_eid = p_eid[sort_idx_t]
        s_cid = p_cids[sort_idx_t]
        s_slot = edge_slots[sort_idx_t]
        
        # Grouping
        MAX_CID = 10000000
        combined_key = s_slot * MAX_CID + s_cid
        unique_keys, counts = torch.unique_consecutive(combined_key, return_counts=True)
        split_points = torch.cumsum(counts, dim=0)[:-1].cpu()
        
        # Tensor Split (View operations, fast)
        t_groups = {
            'src': torch.tensor_split(s_src, split_points),
            'dst': torch.tensor_split(s_dst, split_points),
            'ts':  torch.tensor_split(s_ts, split_points),
            'eid': torch.tensor_split(s_eid, split_points),
            'cid': torch.tensor_split(s_cid, split_points)
        }
        
        unique_keys_cpu = unique_keys.cpu().numpy()
        
        # Submit IO Tasks
        for i, key in enumerate(unique_keys_cpu):
            tid = int(key // MAX_CID)
            cid = int(key % MAX_CID)
            save_path = part_dir / f"slot_{tid:04d}_sub_{cid:06d}.pt"
            
            # Prepare dictionary (Move to CPU here)
            chunk = {
                "src": t_groups['src'][i].cpu(),
                "dst": t_groups['dst'][i].cpu(),
                "ts":  t_groups['ts'][i].cpu(),
                "eid": t_groups['eid'][i].cpu(),
                "cid": t_groups['cid'][i].cpu(),
                "slot_id": tid,
                "cluster_id": cid
            }
            # Async submit
            futures.append(io_pool.submit(_save_chunk_task, chunk, save_path))
            
            # 定期清理已完成任务，防止内存积压
            if len(futures) > 5000:
                futures = [f for f in futures if not f.done()]

    # Wait for all IO
    print("  Waiting for IO to finish...")
    io_pool.shutdown(wait=True)


# ==============================================================================
# Part 4 & 5 (保持基本逻辑，增加进度条和简单优化)
# ==============================================================================

def prepare_distributed_metadata(node_parts, edge_index, num_parts, output_dir):
    print(f"\n[Step 4] Generating Partition Book...")
    src, dst = edge_index
    edge_parts = node_parts[dst]
    partition_book = []
    
    # 简单循环即可，这步很快
    for pid in tqdm(range(num_parts)):
        owned = torch.nonzero(node_parts == pid, as_tuple=True)[0]
        
        mask_p = (edge_parts == pid)
        p_src = src[mask_p]
        # 只在需要的子集上计算 owners
        src_owners = node_parts[p_src] 
        mask_halo = (src_owners != pid)
        halos = torch.unique(p_src[mask_halo]) # GPU unique is fast
        
        partition_book.append(torch.cat([owned, halos]))
    
    torch.save((partition_book, node_parts, edge_parts), output_dir / "partition_book.pt")
    
    try:
        rep_table = build_replica_table(len(node_parts), partition_book, num_parts)
    except:
        rep_table = None
    torch.save(rep_table, output_dir / "replica_table.pt")
    return partition_book, rep_table

def save_distributed_context(output_dir, num_parts, partition_book, edge_owner_part, **kwargs):
    print(f"\n[Step 5] Saving Distributed Features...")
    
    def _slice(data, idx):
        if data is None: return None
        if isinstance(data, list): return [d[idx] for d in data]
        return data[idx]

    # Pre-calculate edge masks to avoid repeated comparison
    edge_masks = []
    has_edge_data = kwargs.get('edge_feat') is not None or kwargs.get('edge_label') is not None
    if has_edge_data:
        for pid in range(num_parts):
            edge_masks.append(edge_owner_part == pid)

    for pid in tqdm(range(num_parts)):
        part_dir = output_dir / f"part_{pid}"
        nodes = partition_book[pid]
        ctx = {}
        
        if kwargs.get('node_feat') is not None: ctx['node_feat'] = _slice(kwargs['node_feat'], nodes)
        if kwargs.get('node_label') is not None: ctx['node_label'] = _slice(kwargs['node_label'], nodes)
        
        if has_edge_data:
            mask = edge_masks[pid]
            if kwargs.get('edge_feat') is not None: ctx['edge_feat'] = _slice(kwargs['edge_feat'], mask)
            if kwargs.get('edge_label') is not None: ctx['edge_label'] = _slice(kwargs['edge_label'], mask)
            
        if ctx: torch.save(ctx, part_dir / "distributed_context.pt")

# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    # 配置
    src_root = Path("/mnt/data/zlj/starrygl-data/ctdg").resolve()
    tgt_root = Path("/mnt/data/zlj/starrygl-data/nparts").resolve()
    num_parts = 4
    hot_ratio = 0.1
    
    # 限制 CPU 线程，避免与 Joblib 冲突
    torch.set_num_threads(4)
    
    for p_path in src_root.glob("*.pth"):
        name = p_path.stem
        if name != "WikiTalk" and name != "StackOverflow": continue 
        
        print(f"=== Processing {name} ===")
        # 使用 map_location 确保加载到 GPU (如果有)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        data = torch.load(p_path, map_location=device)
        num_nodes = data['num_nodes']
        ds = data['dataset']
        
        # 1. 提取全图 (保持在 GPU)
        if isinstance(ds, dict):
            edge_index = ds['edge_index'].to(device)
            edge_ts = ds.get('edge_ts')
            if edge_ts is None and edge_index.shape[0] > 2:
                edge_ts = edge_index[2]
                edge_index = edge_index[:2]
            edge_ts = edge_ts.to(device)
            
            node_feat = ds.get('node_feat')
            node_label = ds.get('y') if 'y' in ds else None
            edge_feat = ds.get('edge_feat')
            edge_label = ds.get('edge_label') if 'edge_label' in ds else None
        else:
            # Handle List[Data]...
            pass 

        # 2. 识别热点 (GPU TopK)
        is_hot = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if hot_ratio > 0:
            deg = torch.bincount(edge_index.flatten(), minlength=num_nodes)
            val, idx = torch.topk(deg, int(num_nodes * hot_ratio))
            is_hot[idx] = True

        # 3. 分区 (Louvain 需要 CPU)
        parts, clusters = partition_hybrid_manifold(num_nodes, edge_index, edge_ts, is_hot, num_parts, part_type='louvain')
        parts = parts.to(device)
        
        # 4. 重排 (并行化)
        # 构造 Dummy Graph，DGL 图构建可能需要 CPU
        g_tmp = dgl.graph((edge_index[0].cpu(), edge_index[1].cpu()), num_nodes=num_nodes)
        avg_ts = torch.zeros(num_nodes, device=device) # Simplify
        perm = hierarchical_spatiotemporal_reordering(g_tmp, parts, is_hot, avg_ts, num_parts)
        perm = perm.to(device)
        
        # 5. 应用映射
        rev_perm = torch.empty_like(perm)
        rev_perm[perm] = torch.arange(num_nodes, device=device)
        
        out_path = tgt_root / f"{name}_{num_parts:03d}"
        out_path.mkdir(parents=True, exist_ok=True)
        torch.save((perm.cpu(), rev_perm.cpu()), out_path / "perm.pt")
        
        # GPU Index Mapping
        new_edge_index = torch.stack([rev_perm[edge_index[0]], rev_perm[edge_index[1]]])
        new_parts = parts[perm]
        new_clusters = clusters[perm.cpu().numpy()]
        
        if node_feat is not None: node_feat = node_feat[perm]
        if node_label is not None: node_label = node_label[perm]
            
        # 6. 生成 Chunks (Async IO)
        prepare_spatiotemporal_chunks(
            new_edge_index, edge_ts, 
            new_parts, new_clusters, 
            num_parts, ("event", 3000), 
            out_path
        )
        
        # 7. 元数据 & 8. 特征 (CPU heavy parts, move data if needed inside func)
        p_book, _ = prepare_distributed_metadata(new_parts, new_edge_index, num_parts, out_path)
        edge_owner = new_parts[new_edge_index[1]]
        save_distributed_context(
            out_path, 
            num_parts, 
            p_book, 
            edge_owner, 
            node_feat=node_feat,   # <--- 关键修改：加上参数名
            node_label=node_label, # <--- 加上参数名
            edge_feat=edge_feat,   # <--- 加上参数名
            edge_label=edge_label  # <--- 加上参数名
        )
        
        print(f"Done {name}.")