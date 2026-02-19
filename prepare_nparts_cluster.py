import heapq
import os
import gc
import time
from pathlib import Path
import pandas as pd
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
    def build_replica_table(num_nodes, partition_book, num_parts): return None

# ==============================================================================
# Utils & Config
# ==============================================================================

class AdaptiveConfig:
    def __init__(self, num_nodes, num_edges, num_parts, num_workers=4, vram_limit_gb=24):
        # --- 1. Micro-Cluster (逻辑锚点) ---
        self.num_micro = int(np.sqrt(num_nodes))
        # [修改建议] 提高下限到 1000，为了更好的 Tetris 负载均衡
        self.num_micro = min(self.num_micro, 5000)
        
        # 保护逻辑：如果图极小(如测试用例)，不能让 micro 数超过节点数的一半
        if self.num_micro > num_nodes // 2:
            self.num_micro = max(1, num_nodes // 2)

        # --- 2. Large-Cluster (物理文件) ---
        BYTES_PER_EDGE = 32  # int64 * 4 (src, dst, ts, eid)
        
        # [约束 A] I/O 效率底线
        # [修改建议] 改回 16MB，1MB 太小了
        MIN_FILE_SIZE = 16 * 1024 * 1024  
        
        # [约束 B] 显存安全顶线
        MAX_FILE_SIZE = 512 * 1024 * 1024 # 512 MB
        
        total_bytes = num_edges * BYTES_PER_EDGE
        
        # 计算允许的文件数量范围
        min_splits_by_vram = int(np.ceil(total_bytes / MAX_FILE_SIZE))
        min_splits_total = max(num_parts, min_splits_by_vram)
        
        max_splits_by_size = int(total_bytes / MIN_FILE_SIZE)
        if max_splits_by_size < min_splits_total:
            max_splits_by_size = min_splits_total
            
        # 计算理想并行数量
        target_parallel = num_parts * num_workers * 2
        
        # 取交集
        self.num_large = 8#min(max(target_parallel, min_splits_total), max_splits_by_size)
        
        # [关键] 对齐：强制是 num_parts 的整数倍 (向上取整)
        if self.num_large % num_parts != 0:
            self.num_large = ((self.num_large // num_parts) + 1) * num_parts

        # 日志输出
        avg_size_mb = total_bytes / self.num_large / 1024 / 1024
        print(f"[Adaptive Config]")
        print(f"  - Data Scale: {num_edges} edges ({total_bytes/1024/1024:.2f} MB)")
        print(f"  - Micro-Clusters: {self.num_micro}")
        print(f"  - Large-Clusters: {self.num_large} (Avg Size: {avg_size_mb:.2f} MB)")
        
        if avg_size_mb > 512:
            print(f"    [WARNING] File size > 512MB, risk of GPU OOM!")

from collections import defaultdict

def evaluate_global_load_balance(edge_tracker, detail_tracker, num_parts):
    """
    全局横向对比报告 (修复版)
    """
    print(f"\n[Step 3.5] Global Load Balance & Micro-Breakdown Report...")
    
    stats_list = []
    # 过滤掉没有任何数据的 TID
    all_tids = sorted([t for t in edge_tracker.keys() if any(edge_tracker[t].values())])
    
    if not all_tids:
        print("  [WARNING] No active time slots found.")
        return

    for tid in all_tids:
        loads = []
        for pid in range(num_parts):
            loads.append(edge_tracker[tid].get(pid, 0))
            
        loads = np.array(loads)
        total_load = loads.sum()
        if total_load == 0: continue
        
        avg_load = loads.mean()
        # 避免除以0
        straggler = loads.max() / (avg_load + 1e-6) if avg_load > 1e-6 else 1.0
        
        stats_list.append({
            "TID": tid,
            "Max_Load": loads.max(),
            "Avg_Load": int(avg_load),
            "Straggler": straggler,
            "Part_Loads": loads.tolist()
        })
        
    if not stats_list: return

    df = pd.DataFrame(stats_list)
    avg_straggler = df['Straggler'].mean()
    max_straggler = df['Straggler'].max()
    
    # 找到最不均衡的时刻
    worst_row = df.loc[df['Straggler'].idxmax()]
    worst_tid = int(worst_row['TID'])
    
    print(f"  > Avg Straggler Ratio: {avg_straggler:.2f}")
    print(f"  > Max Straggler Ratio: {max_straggler:.2f} (at TID {worst_tid})")
    print(f"    - Partition Totals: {worst_row['Part_Loads']}")
    
    # 打印 Worst-Case 的详细微观分布
    print(f"\n  [Micro-Breakdown at Worst TID {worst_tid}]")
    for pid in range(num_parts):
        details = detail_tracker[worst_tid].get(pid, {})
        # 排序以便观察
        sorted_details = dict(sorted(details.items(), key=lambda item: item[1], reverse=True))
        
        if sorted_details:
            top_lcid = next(iter(sorted_details))
            top_load = sorted_details[top_lcid]
            total = sum(sorted_details.values())
            ratio = top_load / total if total > 0 else 0
            # 只显示 Top 3 文件
            short_details = dict(list(sorted_details.items())[:3])
            print(f"    - Part {pid}: Total={total} | Top File (LCID {top_lcid}) = {top_load} ({ratio:.1%})")
            if len(sorted_details) > 1:
                print(f"      -> Top 3 Files: {short_details} ...")
        else:
            print(f"    - Part {pid}: (Idle)")
import pandas as pd
import numpy as np
from collections import defaultdict

def evaluate_global_load_balance(edge_tracker, detail_tracker, num_parts, output_dir=None):
    """
    [增强版] 全局负载评估：输出所有 TID 下每个 Partition 的详细负载
    """
    print(f"\n[Step 3.5] Global Load Balance & Micro-Breakdown Report...")
    
    # 1. 准备数据容器
    summary_stats = []
    detailed_rows = [] # 用于生成 CSV
    
    # 过滤掉全空的 TID
    all_tids = sorted([t for t in edge_tracker.keys() if any(edge_tracker[t].values())])
    
    if not all_tids:
        print("  [WARNING] No active time slots found.")
        return

    print(f"  {'TID':<6} | {'Total':<8} | {'Avg/Part':<8} | {'Straggler':<9} | {'Partition Loads (Edge Counts)':<40}")
    print("-" * 90)

    for tid in all_tids:
        loads = []
        for pid in range(num_parts):
            count = edge_tracker[tid].get(pid, 0)
            loads.append(count)
            
            # --- 收集详细信息用于 CSV ---
            # 找出该分区内部最忙的文件 (Top-1 Large Cluster)
            p_details = detail_tracker[tid].get(pid, {})
            if p_details:
                top_lcid = max(p_details, key=p_details.get)
                top_load = p_details[top_lcid]
            else:
                top_lcid = -1
                top_load = 0
                
            detailed_rows.append({
                "TID": tid,
                "Partition": pid,
                "Total_Edges": count,
                "Top1_File_ID": top_lcid,
                "Top1_File_Load": top_load,
                "Load_Ratio": top_load / count if count > 0 else 0
            })
            # ---------------------------

        loads = np.array(loads)
        total_load = loads.sum()
        if total_load == 0: continue
        
        avg_load = loads.mean()
        # Straggler Ratio
        straggler = loads.max() / (avg_load + 1e-6) if avg_load > 1e-6 else 1.0
        
        summary_stats.append(straggler)
        
        # 格式化打印每一行 TID 的情况
        # 例如: TID 0 | 1000 | 250 | 1.0 | [250, 250, 250, 250]
        load_str = str(loads.tolist())
        print(f"  {tid:<6} | {total_load:<8} | {int(avg_load):<8} | {straggler:<9.2f} | {load_str}")

    # 2. 打印总体摘要
    avg_straggler = np.mean(summary_stats) if summary_stats else 0
    max_straggler = np.max(summary_stats) if summary_stats else 0
    
    print("-" * 90)
    print(f"  > Overall Avg Straggler: {avg_straggler:.2f}")
    print(f"  > Overall Max Straggler: {max_straggler:.2f}")

    # 3. 保存详细 CSV (如果提供了 output_dir)
    if output_dir:
        csv_path = f"{output_dir}/load_balance_details.csv"
        df_detail = pd.DataFrame(detailed_rows)
        df_detail.to_csv(csv_path, index=False)
        print(f"  > Full details saved to: {csv_path}")
        
    # 4. 简单的瓶颈诊断建议
    if avg_straggler > 1.5:
        print(f"  [DIAGNOSIS] High Imbalance detected.")
        print(f"  -> Check the CSV to see which Partition is overloaded.")
def prepare_spatiotemporal_chunks_v2(
    edge_index_cpu, edge_ts_cpu, 
    node_parts_cpu, node_clusters_cpu, 
    micro_to_large, 
    num_parts, slice_param, output_dir
):
    print(f"\n[Step 3] Generating Monolithic Data & Analyzing Global Load...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    src_np = edge_index_cpu[0].numpy()
    dst_np = edge_index_cpu[1].numpy()
    ts_np = edge_ts_cpu.numpy()
    node_parts_np = node_parts_cpu.numpy()
    
    # 时间边界计算
    if slice_param[0] == "event":
        u_ts, counts = np.unique(ts_np, return_counts=True)
        cum = np.cumsum(counts)
        boundaries = [ts_np.min()]
        curr, total = 0, cum[-1]
        while curr < total:
            target = curr + slice_param[1]
            if target >= total: break
            idx = np.searchsorted(cum, target)
            boundaries.append(u_ts[idx])
            curr = cum[idx]
        if boundaries[-1] < ts_np.max(): boundaries.append(ts_np.max() + 1)
        time_boundaries = np.array(boundaries)
    else:
        time_boundaries = np.linspace(ts_np.min(), ts_np.max() + 1, slice_param[1])

    torch.save({"boundaries": torch.from_numpy(time_boundaries), "strategy": slice_param}, output_dir / "dist_meta.pt")
    
    # [NEW] 全局追踪器
    global_edge_tracker = defaultdict(dict) 
    global_detail_tracker = defaultdict(lambda: defaultdict(dict))
    
    edge_owners = node_parts_np[dst_np]
    
    for pid in tqdm(range(num_parts), desc="Partitions"):
        part_dir = output_dir / f"part_{pid}"
        part_dir.mkdir(exist_ok=True)
        
        mask = (edge_owners == pid)
        p_src, p_dst, p_ts, p_eid = src_np[mask], dst_np[mask], ts_np[mask], np.arange(len(src_np))[mask]
        if len(p_src) == 0: continue

        edge_slots = np.searchsorted(time_boundaries, p_ts, side='right') - 1
        valid_mask = (edge_slots >= 0) & (edge_slots < len(time_boundaries) - 1)
        p_src, p_dst, p_ts, p_eid, edge_slots = p_src[valid_mask], p_dst[valid_mask], p_ts[valid_mask], p_eid[valid_mask], edge_slots[valid_mask]
        
        p_micro = node_clusters_cpu[p_dst]
        p_large = micro_to_large[p_micro] 
        
        # 排序
        sort_idx = np.lexsort((p_ts, p_large, edge_slots)) 
        s_src, s_dst, s_ts, s_eid = p_src[sort_idx], p_dst[sort_idx], p_ts[sort_idx], p_eid[sort_idx]
        s_large, s_slot = p_large[sort_idx], edge_slots[sort_idx]
        
        # 构建索引
        MAX_VAL = 1000000
        combined_key = s_slot.astype(np.int64) * MAX_VAL + s_large
        unique_keys, idx_start, counts = np.unique(combined_key, return_index=True, return_counts=True)
        
        data_index = {}
        for i, key in enumerate(unique_keys):
            tid = int(key // MAX_VAL)
            lcid = int(key % MAX_VAL)
            count = int(counts[i])
            start = int(idx_start[i])
            
            if tid not in data_index: data_index[tid] = {}
            data_index[tid][lcid] = (start, count)
            
            # [TRACKING] 记录到全局追踪器
            current_pid_load = global_edge_tracker[tid].get(pid, 0)
            global_edge_tracker[tid][pid] = current_pid_load + count
            global_detail_tracker[tid][pid][lcid] = count

        # [DELETE] 删掉了原来的 evaluate_load_balance 调用，因为它容易报错且视角局限

        torch.save({
            "src": torch.from_numpy(s_src),
            "dst": torch.from_numpy(s_dst),
            "ts":  torch.from_numpy(s_ts),
            "eid": torch.from_numpy(s_eid),
            "cid": torch.from_numpy(s_large),
            "slot_id": torch.from_numpy(s_slot),
            "index": data_index
        }, part_dir / "data.pt")
        
    print("  Partitioning complete.")
    
    # [NEW] 统一进行全局评估
    #evaluate_global_load_balance(global_edge_tracker, global_detail_tracker, num_parts)
    # [NEW] 传入 output_dir 用于保存 CSV
    evaluate_global_load_balance(global_edge_tracker, global_detail_tracker, num_parts, '.')
# ==============================================================================
# Part 1: Partitioning & Aggregation
# ==============================================================================

def vector_balanced_aggregation_partition_aware(
    node_clusters_raw, edge_index_cpu, edge_ts_cpu, 
    num_large_clusters, node_parts_cpu, 
    num_parts, 
    num_time_bins=128
):
    """
    [回归简单] Partition-Aware Mean/Std Round-Robin
    """
    print(f"    - Running Partition-Aware Round-Robin Aggregation...")
    
    dst_np = edge_index_cpu[1].numpy()
    ts_np = edge_ts_cpu.numpy()
    node_parts_np = node_parts_cpu.numpy()
    max_micro = node_clusters_raw.max() + 1
    
    # 1. 确定 Micro -> Partition 映射
    df_map = pd.DataFrame({'cid': node_clusters_raw, 'pid': node_parts_np})
    micro_to_part = df_map.drop_duplicates('cid').set_index('cid')['pid'].reindex(range(max_micro), fill_value=-1).values
    
    # 2. 计算时间特征 (Mean)
    print("      -> Calculating stats for aggregation...")
    edge_micro_cids = node_clusters_raw[dst_np]
    
    cluster_counts = np.bincount(edge_micro_cids, minlength=max_micro)
    cluster_time_sum = np.bincount(edge_micro_cids, weights=ts_np, minlength=max_micro)
    
    means = np.zeros(max_micro)
    valid_mask = cluster_counts > 0
    means[valid_mask] = cluster_time_sum[valid_mask] / cluster_counts[valid_mask]
    
    # 3. 分区内轮询分配
    final_micro_to_large = np.full(max_micro, -1, dtype=np.int32)
    large_per_part = max(1, num_large_clusters // num_parts) # 防止除0
    
    for pid in range(num_parts):
        # 找出该分区的 Micro Clusters
        part_micro_indices = np.where((micro_to_part == pid) & (cluster_counts > 0))[0]
        if len(part_micro_indices) == 0: continue
        
        # 按时间排序
        local_means = means[part_micro_indices]
        sorted_indices = part_micro_indices[np.argsort(local_means)]
        
        # 轮询分配给 Large Cluster
        # 这确保了 part_0 的 large_0, large_1, large_2 也是按时间轮流拿数据的
        start_lid = pid * large_per_part
        for rank, m_cid in enumerate(sorted_indices):
            local_lid = rank % large_per_part
            final_micro_to_large[m_cid] = start_lid + local_lid
            
    # 孤儿处理
    orphan_indices = np.where(final_micro_to_large == -1)[0]
    for m_cid in orphan_indices:
        pid = micro_to_part[m_cid]
        if pid == -1: pid = 0 # 防御性编程
        local_lid = m_cid % large_per_part
        final_micro_to_large[m_cid] = pid * large_per_part + local_lid

    return final_micro_to_large
# def vector_balanced_aggregation_partition_aware(
#     node_clusters_raw, edge_index_cpu, edge_ts_cpu, 
#     num_large_clusters, node_parts_cpu, 
#     num_parts, 
#     num_time_bins=128
# ):
#     """
#     [融合策略] Partition-Aware Vector Tetris Packing
#     """
#     print(f"    - Running Partition-Aware Vector Balancing (Tetris Strategy)...")
    
#     dst_np = edge_index_cpu[1].numpy()
#     ts_np = edge_ts_cpu.numpy()
#     node_parts_np = node_parts_cpu.numpy()
#     max_micro = node_clusters_raw.max() + 1
    
#     # 1. 确定 Micro -> Partition 映射
#     df_map = pd.DataFrame({'cid': node_clusters_raw, 'pid': node_parts_np})
#     micro_to_part = df_map.drop_duplicates('cid').set_index('cid')['pid'].reindex(range(max_micro), fill_value=-1).values
    
#     # 2. 构建负载画像
#     print("      -> Building load profiles & stats...")
#     t_min, t_max = ts_np.min(), ts_np.max()
#     edge_micro_cids = node_clusters_raw[dst_np]
    
#     ts_bins = np.floor((ts_np - t_min) / (t_max - t_min + 1e-6) * num_time_bins).astype(np.int32)
#     combined_key = edge_micro_cids.astype(np.int64) * num_time_bins + ts_bins
#     u_keys, counts = np.unique(combined_key, return_counts=True)
    
#     micro_profiles = np.zeros((max_micro, num_time_bins), dtype=np.int32)
#     micro_profiles[u_keys // num_time_bins, u_keys % num_time_bins] = counts
    
#     # 手动计算加权平均 (避免 np.average shape 报错)
#     bin_indices = np.arange(num_time_bins)
#     weighted_sum = np.sum(bin_indices * micro_profiles, axis=1) # Broadcast works here: (128,) * (N, 128) -> (N, 128)
#     sum_of_weights = np.sum(micro_profiles, axis=1) + 1e-6
#     micro_mean = weighted_sum / sum_of_weights
    
#     micro_total_load = micro_profiles.sum(axis=1)
    
#     # 3. 核心装箱循环
#     final_micro_to_large = np.full(max_micro, -1, dtype=np.int32)
#     large_per_part = num_large_clusters // num_parts
    
#     for pid in range(num_parts):
#         part_micro_indices = np.where((micro_to_part == pid) & (micro_total_load > 0))[0]
#         if len(part_micro_indices) == 0: continue
        
#         # 排序：优先负载大的，其次按时间均值
#         local_loads = micro_total_load[part_micro_indices]
#         local_means = micro_mean[part_micro_indices]
#         sort_keys = np.lexsort((local_means, -local_loads)) 
#         valid_indices = part_micro_indices[sort_keys]
        
#         start_lid = pid * large_per_part
#         local_large_profiles = np.zeros((large_per_part, num_time_bins), dtype=np.int32)
#         local_large_peaks = np.zeros(large_per_part, dtype=np.int32)
        
#         for m_cid in valid_indices:
#             m_vec = micro_profiles[m_cid]
            
#             best_local_lid = -1
#             min_peak_increase = float('inf')
#             best_new_peak = float('inf')
            
#             for local_lid in range(large_per_part):
#                 curr_peak = local_large_peaks[local_lid]
#                 new_profile = local_large_profiles[local_lid] + m_vec
#                 new_peak = np.max(new_profile)
#                 increase = new_peak - curr_peak
                
#                 if increase < min_peak_increase:
#                     min_peak_increase = increase
#                     best_new_peak = new_peak
#                     best_local_lid = local_lid
#                 elif increase == min_peak_increase:
#                     if new_peak < best_new_peak:
#                         best_new_peak = new_peak
#                         best_local_lid = local_lid
            
#             if best_local_lid != -1:
#                 global_lid = start_lid + best_local_lid
#                 final_micro_to_large[m_cid] = global_lid
#                 local_large_profiles[best_local_lid] += m_vec
#                 local_large_peaks[best_local_lid] = best_new_peak

#     # 4. 孤儿处理
#     orphan_mask = (final_micro_to_large == -1) & (micro_to_part != -1)
#     orphan_indices = np.where(orphan_mask)[0]
#     for m_cid in orphan_indices:
#         pid = micro_to_part[m_cid]
#         local_lid = m_cid % large_per_part
#         final_micro_to_large[m_cid] = pid * large_per_part + local_lid

#     return final_micro_to_large

metis = dgl.distributed.partition.metis_partition_assignment
def apply_inter_partition(g: dgl.DGLGraph, node_types: torch.Tensor | None, k: int) -> torch.Tensor:
    print("Enter apply_inter_partition()")
    node_parts: torch.Tensor = metis(g, k=k, balance_ntypes=node_types, balance_edges=True)
    assert node_parts.max().item() + 1 == k, f"node_parts.max().item() + 1 != k"
    return node_parts.type(torch.uint8)

def partition_hybrid_manifold_fused(
    num_nodes, edge_index, edge_ts, hot_mask, num_parts, 
    num_anchors=1000, 
    num_time_bins=128 # 这个参数在这里没用了，但为了接口兼容保留
):
    print(f"[Step 1] Running Two-Stage Partitioning (Mean/Std + Round Robin)...")
    
    # --- Stage 1: Metis Micro-Partitioning ---
    src_t = edge_index[0].cpu()
    dst_t = edge_index[1].cpu()
    ts_np = edge_ts.cpu().numpy()

    print(f"    - Phase A: Metis micro-clustering (K={num_anchors})...")
    g = dgl.graph((src_t, dst_t), num_nodes=num_nodes)
    node_parts_tensor = apply_inter_partition(g, None, k=num_anchors)
    node_to_cluster = node_parts_tensor.numpy().astype(np.int32)
    max_cluster_id = max(num_anchors, node_to_cluster.max() + 1)

    # --- Select Anchors ---
    print("    - Selecting Anchors...")
    deg = (g.in_degrees() + g.out_degrees()).numpy()
    df = pd.DataFrame({'nid': np.arange(num_nodes), 'cid': node_to_cluster, 'deg': deg})
    best_idx = df.groupby('cid')['deg'].idxmax()
    anchors_old = np.zeros(max_cluster_id, dtype=np.int64)
    valid_cids = best_idx.index.values
    anchors_old[valid_cids] = df.loc[best_idx, 'nid'].values

    # --- Stage 2: Time-Feature Based Round Robin ---
    print(f"    - Phase B: Assigning {max_cluster_id} micro-clusters to {num_parts} partitions (Round-Robin)...")
    
    dst_np = dst_t.numpy()
    edge_clusters = node_to_cluster[dst_np]
    
    # 1. 计算每个 Micro-Cluster 的时间均值 (Mean) 和标准差 (Std)
    print("      -> Calculating temporal features...")
    
    # 使用 bincount 快速计算 sum 和 count
    cluster_counts = np.bincount(edge_clusters, minlength=max_cluster_id)
    cluster_time_sum = np.bincount(edge_clusters, weights=ts_np, minlength=max_cluster_id)
    # 计算平方和用于 std (E[X^2])
    cluster_time_sq_sum = np.bincount(edge_clusters, weights=ts_np**2, minlength=max_cluster_id)
    
    # 防止除以0
    valid_mask = cluster_counts > 0
    
    means = np.zeros(max_cluster_id)
    stds = np.zeros(max_cluster_id)
    
    if np.any(valid_mask):
        means[valid_mask] = cluster_time_sum[valid_mask] / cluster_counts[valid_mask]
        # Var = E[X^2] - (E[X])^2
        avg_sq = cluster_time_sq_sum[valid_mask] / cluster_counts[valid_mask]
        vars = avg_sq - means[valid_mask]**2
        vars = np.maximum(vars, 0) # 修正浮点误差
        stds[valid_mask] = np.sqrt(vars)
        
    # 2. 排序策略
    # 优先按 Mean 排序 (时间轴顺序)，其次按 Std (持续时长)
    # 对于没有边的 Cluster (count=0)，它们会被排在最前面或最后面，不影响
    sort_keys = np.lexsort((stds, means)) # keys: Primary=means, Secondary=stds
    
    # 3. 强制轮询分配 (Round Robin)
    # 这保证了时间相近的 Cluster 被均匀打散到不同机器
    # Cluster 0 (T=0) -> Part 0
    # Cluster 1 (T=1) -> Part 1
    # ...
    cluster_to_part = np.zeros(max_cluster_id, dtype=np.int32)
    
    # 只需要对有效 Cluster 进行分配，无效的随便
    valid_sorted_indices = [idx for idx in sort_keys if cluster_counts[idx] > 0]
    
    for rank, cid in enumerate(valid_sorted_indices):
        cluster_to_part[cid] = rank % num_parts
        
    # 孤儿分配
    orphan_indices = np.where(cluster_counts == 0)[0]
    for i, cid in enumerate(orphan_indices):
        cluster_to_part[cid] = i % num_parts

    final_parts = torch.from_numpy(cluster_to_part[node_to_cluster]).long()
    
    # 简单的负载检查
    part_counts = np.bincount(cluster_to_part, weights=cluster_counts, minlength=num_parts)
    print(f"    - [Partition Balance Report]")
    print(f"      -> Total Load Imbalance: {(part_counts.max() / part_counts.mean()):.4f}")
    
    return final_parts, node_to_cluster, torch.from_numpy(anchors_old)

# def partition_hybrid_manifold_fused(
#     num_nodes, edge_index, edge_ts, hot_mask, num_parts, 
#     num_anchors=1000, 
#     num_time_bins=128
# ):
#     print(f"[Step 1] Running Two-Stage Partitioning (Metis -> Tetris Packing)...")
#     src_t = edge_index[0].cpu()
#     dst_t = edge_index[1].cpu()
#     ts_np = edge_ts.cpu().numpy()

#     # --- Stage 1: Metis Micro-Partitioning ---
#     print(f"    - Phase A: Metis micro-clustering (K={num_anchors})...")
#     g = dgl.graph((src_t, dst_t), num_nodes=num_nodes)
#     node_parts_tensor = apply_inter_partition(g, None, k=num_anchors)
#     node_to_cluster = node_parts_tensor.numpy().astype(np.int32)
#     max_cluster_id = max(num_anchors, node_to_cluster.max() + 1)

#     # --- Select Anchors ---
#     print("    - Selecting Anchors...")
#     deg = (g.in_degrees() + g.out_degrees()).numpy()
#     df = pd.DataFrame({'nid': np.arange(num_nodes), 'cid': node_to_cluster, 'deg': deg})
#     best_idx = df.groupby('cid')['deg'].idxmax()
#     anchors_old = np.zeros(max_cluster_id, dtype=np.int64)
#     valid_cids = best_idx.index.values
#     anchors_old[valid_cids] = df.loc[best_idx, 'nid'].values

#     # --- Stage 2: Tetris Packing (Partition Level) ---
#     print(f"    - Phase B: Assigning {max_cluster_id} micro-clusters to {num_parts} partitions...")
    
#     dst_np = dst_t.numpy()
#     edge_clusters = node_to_cluster[dst_np]
    
#     t_min, t_max = ts_np.min(), ts_np.max()
#     ts_bins = np.floor((ts_np - t_min) / (t_max - t_min + 1e-6) * num_time_bins).astype(np.int32)
    
#     combined_key = edge_clusters.astype(np.int64) * num_time_bins + ts_bins
#     u_keys, counts = np.unique(combined_key, return_counts=True)
    
#     micro_profiles = np.zeros((max_cluster_id, num_time_bins), dtype=np.int32)
#     micro_profiles[u_keys // num_time_bins, u_keys % num_time_bins] = counts
    
#     micro_total_load = micro_profiles.sum(axis=1)
#     micro_peak_load = micro_profiles.max(axis=1)
    
#     priority = micro_total_load + micro_peak_load * 2.0
#     sorted_micro_ids = np.argsort(-priority)
#     valid_micro_ids = [cid for cid in sorted_micro_ids if micro_total_load[cid] > 0]
    
#     part_profiles = np.zeros((num_parts, num_time_bins), dtype=np.int32)
#     part_peaks = np.zeros(num_parts, dtype=np.int32)
#     cluster_to_part = np.zeros(max_cluster_id, dtype=np.int32)
#     orphan_ptr = 0
    
#     for cid in tqdm(valid_micro_ids, desc="      -> Tetris Packing"):
#         vec = micro_profiles[cid]
#         best_pid = -1
#         min_peak_increase = float('inf')
#         best_new_peak = float('inf')
        
#         for pid in range(num_parts):
#             curr_profile = part_profiles[pid]
#             curr_peak = part_peaks[pid]
#             new_profile = curr_profile + vec
#             new_peak = np.max(new_profile)
#             increase = new_peak - curr_peak
            
#             if increase < min_peak_increase:
#                 min_peak_increase = increase
#                 best_new_peak = new_peak
#                 best_pid = pid
#             elif increase == min_peak_increase:
#                 if new_peak < best_new_peak:
#                     best_new_peak = new_peak
#                     best_pid = pid
        
#         cluster_to_part[cid] = best_pid
#         part_profiles[best_pid] += vec
#         part_peaks[best_pid] = best_new_peak
        
#     orphan_indices = np.where(micro_total_load == 0)[0]
#     for cid in orphan_indices:
#         cluster_to_part[cid] = orphan_ptr % num_parts
#         orphan_ptr += 1

#     final_parts = torch.from_numpy(cluster_to_part[node_to_cluster]).long()
    
#     final_loads = part_profiles.sum(axis=1)
#     print("    - [Partition Balance Report]")
#     print(f"      -> Total Load Imbalance: {(final_loads.max()/final_loads.mean()):.4f}")
    
#     return final_parts, node_to_cluster, torch.from_numpy(anchors_old)

# ==============================================================================
# Part 2: Reordering (并行化优化)
# ==============================================================================

def _process_single_partition_rcm(pid, p_nodes, full_adj_indices, full_adj_indptr, hot_mask_np, node_avg_ts_np, num_time_buckets):
    if len(p_nodes) == 0: return np.array([], dtype=np.int64)
    is_hot = hot_mask_np[p_nodes]
    local_hubs = p_nodes[is_hot]
    local_cold = p_nodes[~is_hot]
    cold_layout = []
    if len(local_cold) > 0:
        times = node_avg_ts_np[local_cold]
        sorted_cold = local_cold[np.argsort(times)]
        buckets = np.array_split(sorted_cold, num_time_buckets)
        for bucket in buckets:
            if len(bucket) == 0: continue
            data = np.ones(len(full_adj_indices), dtype=np.int8) 
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
    
    try: adj = graph.adj_external(scipy_fmt='csr')
    except: adj = graph.adj(scipy_fmt='csr')
    indices, indptr = adj.indices, adj.indptr
    
    results = Parallel(n_jobs=min(num_parts, 16), backend="loky")(
        delayed(_process_single_partition_rcm)(
            pid, np.where(node_parts_np == pid)[0], indices, indptr, 
            hot_mask_np, node_avg_ts_np, num_time_buckets
        ) for pid in range(num_parts)
    )
    full_perm = np.concatenate(results)
    if len(full_perm) != graph.num_nodes():
        mask = np.ones(graph.num_nodes(), dtype=bool)
        mask[full_perm] = False
        full_perm = np.concatenate([full_perm, np.where(mask)[0]])
    return torch.from_numpy(full_perm)

# ==============================================================================
# Part 3: Chunk Generation & Saving
# ==============================================================================

def map_anchors_and_save(anchors_old, node_to_cluster_raw, perm_cpu, num_nodes, output_dir):
    print(f"\n[Step 2.5] Mapping Anchors to New ID Space...")
    anchors_new = perm_cpu[anchors_old]
    torch.save(anchors_new, output_dir / "anchors.pt")
    print(f"    - Anchors Mapped & Saved. Count: {len(anchors_new)}")
    return anchors_new

# def prepare_spatiotemporal_chunks_v2(
#     edge_index_cpu, edge_ts_cpu, 
#     node_parts_cpu, node_clusters_cpu, 
#     micro_to_large, 
#     num_parts, slice_param, output_dir
# ):
#     print(f"\n[Step 3] Generating Monolithic Data (Simplified: No Micro-CIDs)...")
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     src_np = edge_index_cpu[0].numpy()
#     dst_np = edge_index_cpu[1].numpy()
#     ts_np = edge_ts_cpu.numpy()
#     node_parts_np = node_parts_cpu.numpy()
    
#     # 时间边界
#     if slice_param[0] == "event":
#         u_ts, counts = np.unique(ts_np, return_counts=True)
#         cum = np.cumsum(counts)
#         boundaries = [ts_np.min()]
#         curr, total = 0, cum[-1]
#         while curr < total:
#             target = curr + slice_param[1]
#             if target >= total: break
#             idx = np.searchsorted(cum, target)
#             boundaries.append(u_ts[idx])
#             curr = cum[idx]
#         if boundaries[-1] < ts_np.max(): boundaries.append(ts_np.max() + 1)
#         time_boundaries = np.array(boundaries)
#     else:
#         time_boundaries = np.linspace(ts_np.min(), ts_np.max() + 1, slice_param[1])

#     torch.save({
#         "boundaries": torch.from_numpy(time_boundaries), 
#         "strategy": slice_param
#     }, output_dir / "dist_meta.pt")
    
#     edge_owners = node_parts_np[dst_np]
    
#     for pid in tqdm(range(num_parts), desc="Partitions"):
#         part_dir = output_dir / f"part_{pid}"
#         part_dir.mkdir(exist_ok=True)
        
#         mask = (edge_owners == pid)
#         p_src, p_dst, p_ts, p_eid = src_np[mask], dst_np[mask], ts_np[mask], np.arange(len(src_np))[mask]
#         if len(p_src) == 0: continue

#         edge_slots = np.searchsorted(time_boundaries, p_ts, side='right') - 1
#         valid_mask = (edge_slots >= 0) & (edge_slots < len(time_boundaries) - 1)
#         p_src, p_dst, p_ts, p_eid, edge_slots = p_src[valid_mask], p_dst[valid_mask], p_ts[valid_mask], p_eid[valid_mask], edge_slots[valid_mask]
        
#         p_micro = node_clusters_cpu[p_dst]
#         p_large = micro_to_large[p_micro] 
        
#         # [排序] Slot -> Large -> Time
#         # lexsort keys: (Tertiary, Secondary, Primary)
#         sort_idx = np.lexsort((p_ts, p_large, edge_slots)) 
        
#         s_src, s_dst, s_ts, s_eid = p_src[sort_idx], p_dst[sort_idx], p_ts[sort_idx], p_eid[sort_idx]
#         s_large, s_slot = p_large[sort_idx], edge_slots[sort_idx]
        
#         # 构建索引
#         MAX_VAL = 1000000
#         combined_key = s_slot.astype(np.int64) * MAX_VAL + s_large
#         unique_keys, idx_start, counts = np.unique(combined_key, return_index=True, return_counts=True)
        
#         data_index = {}
#         for i, key in enumerate(unique_keys):
#             tid = int(key // MAX_VAL)
#             lcid = int(key % MAX_VAL)
#             if tid not in data_index: data_index[tid] = {}
#             data_index[tid][lcid] = (int(idx_start[i]), int(counts[i]))
            
#         # 负载评估
#         num_large_estimated = micro_to_large.max() + 1
#         evaluate_load_balance(data_index, pid, num_large_estimated)
        
#         # 落盘
#         torch.save({
#             "src": torch.from_numpy(s_src),
#             "dst": torch.from_numpy(s_dst),
#             "ts":  torch.from_numpy(s_ts),
#             "eid": torch.from_numpy(s_eid),
#             "cid": torch.from_numpy(s_large), # "large_cid"
#             "slot_id": torch.from_numpy(s_slot),
#             "index": data_index
#         }, part_dir / "data.pt")
        
#     print("  Partitioning complete (Minimalist Mode).")

def prepare_distributed_metadata(node_parts, edge_index, num_parts, output_dir, anchor_nodes=None):
    print(f"\n[Step 4] Generating Partition Book (Ordered)...")
    src, dst = edge_index
    edge_parts = node_parts[dst]
    partition_book = []
    
    if anchor_nodes is None: anchor_nodes = torch.empty(0, dtype=torch.long)
    anchor_nodes_np = anchor_nodes.cpu().numpy()

    for pid in tqdm(range(num_parts)):
        owned = torch.nonzero(node_parts == pid, as_tuple=True)[0]
        mask_p = (edge_parts == pid)
        p_src = src[mask_p]
        src_owners = node_parts[p_src] 
        mask_halo = (src_owners != pid)
        halos = torch.unique(p_src[mask_halo]) 
        combined_np = np.concatenate([owned.numpy(), halos.numpy(), anchor_nodes_np])
        _, idx = np.unique(combined_np, return_index=True)
        final_nodes_np = combined_np[np.sort(idx)]
        partition_book.append(torch.from_numpy(final_nodes_np))
    
    torch.save((partition_book, node_parts, edge_parts), output_dir / "partition_book.pt")
    try: rep_table = build_replica_table(len(node_parts), partition_book, num_parts)
    except: rep_table = None
    rep_table = {'indptr':rep_table.indptr, 'indices':rep_table.indices, 'locs':rep_table.locs} if rep_table is not None else None
    torch.save(rep_table, output_dir / "replica_table.pt")
    return partition_book, rep_table

def save_distributed_context(output_dir, num_parts, partition_book, edge_owner_part, **kwargs):
    print(f"\n[Step 5] Saving Distributed Features...")
    def _slice(data, idx):
        if data is None: return None
        if isinstance(data, list): return [d[idx] for d in data]
        return data[idx]

    has_edge_data = kwargs.get('edge_feat') is not None or kwargs.get('edge_label') is not None
    edge_masks = []
    if has_edge_data:
        for pid in range(num_parts): edge_masks.append(edge_owner_part == pid)

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

if __name__ == "__main__":
    src_root = Path("/mnt/data/zlj/starrygl-data/ctdg").resolve()
    tgt_root = Path("/mnt/data/zlj/starrygl-data/nparts").resolve()
    num_parts = 4
    hot_ratio = 0.1
    
    torch.set_num_threads(8)
    
    for p_path in src_root.glob("*.pth"):
        name = p_path.stem
        #if name != 'WIKI' and name != 'StackOverflow' and 
        if name != 'WikiTalk': continue
        print(f"=== Processing {name} ===")
        
        data = torch.load(p_path, map_location='cpu')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_nodes = data['num_nodes']
        ds = data['dataset']
        
        if isinstance(ds, dict):
            edge_index = ds['edge_index'].to(device, non_blocking=True)
            edge_ts = ds.get('edge_ts')
            if edge_ts is None and edge_index.shape[0] > 2:
                edge_ts = edge_index[2]
                edge_index = edge_index[:2]
            if edge_ts is not None: edge_ts = edge_ts.to(device, non_blocking=True)
            node_feat = ds.get('node_feat')
            node_label = ds.get('y') if 'y' in ds else None
            edge_feat = ds.get('edge_feat')
            edge_label = ds.get('edge_label') if 'edge_label' in ds else None
        
        is_hot = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if hot_ratio > 0:
            deg = torch.bincount(edge_index.flatten(), minlength=num_nodes)
            val, idx = torch.topk(deg, int(num_nodes * hot_ratio))
            is_hot[idx] = True

        # [NEW] 自适应计算参数
        adaptive_cfg = AdaptiveConfig(num_nodes, edge_index.shape[1], num_parts)

        # 3. [Fused] 分区 & 锚点生成 (使用自适应 K)
        parts, clusters_raw, anchors_old = partition_hybrid_manifold_fused(
            num_nodes, edge_index, edge_ts, is_hot, num_parts, 
            num_anchors=adaptive_cfg.num_micro
        )
        parts = parts.to(device)
        anchors_old = anchors_old.cpu()
        
        # 4. 重排
        g_tmp = dgl.graph((edge_index[0].cpu(), edge_index[1].cpu()), num_nodes=num_nodes)
        avg_ts = torch.zeros(num_nodes, device=device) 
        perm = hierarchical_spatiotemporal_reordering(g_tmp, parts, is_hot, avg_ts, num_parts)
        perm = perm.to(device)
        
        out_path = tgt_root / f"{name}_{num_parts:03d}"
        out_path.mkdir(parents=True, exist_ok=True)
        
        perm_cpu = perm.cpu()
        rev_perm_cpu = torch.empty(num_nodes, dtype=torch.long)
        rev_perm_cpu[perm_cpu] = torch.arange(num_nodes)
        torch.save((perm_cpu, rev_perm_cpu), out_path / "perm.pt")
        
        # 5. [Mapped] 锚点映射
        anchors_new = map_anchors_and_save(anchors_old, clusters_raw, perm_cpu, num_nodes, out_path)

        print("Moving to CPU...")
        edge_index_cpu = edge_index.cpu()
        parts_cpu = parts.cpu() 
        edge_ts_cpu = edge_ts.cpu()
        del edge_index, parts, edge_ts, g_tmp, is_hot, perm
        torch.cuda.empty_cache() 
        gc.collect()

        print("Re-indexing on CPU...")
        new_src = rev_perm_cpu[edge_index_cpu[0]]
        new_dst = rev_perm_cpu[edge_index_cpu[1]]
        new_edge_index = torch.stack([new_src, new_dst]) 
        new_parts = parts_cpu[perm_cpu]
        new_clusters = clusters_raw[perm_cpu.numpy()] 
        
        if node_feat is not None: 
            node_feat = node_feat.cpu(); node_feat = node_feat[perm_cpu] 
        if node_label is not None:
            node_label = node_label.cpu(); node_label = node_label[perm_cpu]

        # [NEW] 5.5: 自适应生成 Large-Cluster 映射 (Vector Tetris)
        micro_to_large = vector_balanced_aggregation_partition_aware(
            new_clusters, new_edge_index, edge_ts_cpu,
            num_large_clusters=adaptive_cfg.num_large, node_parts_cpu=new_parts,
            num_parts=num_parts, num_time_bins=128
        )

        # 6. 生成 Monolithic Files
        prepare_spatiotemporal_chunks_v2(
            new_edge_index, edge_ts_cpu, 
            new_parts, new_clusters, 
            micro_to_large,
            num_parts, ("event", 12000), 
            out_path
        )
        
        # 7. 元数据
        p_book, _ = prepare_distributed_metadata(new_parts, new_edge_index, num_parts, out_path, anchor_nodes=anchors_new)
        edge_owner = new_parts[new_edge_index[1]]
        save_distributed_context(out_path, num_parts, p_book, edge_owner, node_feat=node_feat, node_label=node_label, edge_feat=edge_feat, edge_label=edge_label)
        
        print(f"Done {name}.")