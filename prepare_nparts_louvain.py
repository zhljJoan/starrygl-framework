import os
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import pymetis
import scipy as sp
import torch
import numpy as np
import dgl
import networkx as nx
import community as community_louvain 
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

# 请确保这些模块在你项目中存在，或者使用 mock 类
from starrygl.cache.cache_route import CommPlan
from starrygl.cache.replica_table import build_replica_table
from starrygl.route.route import Route

# ==============================================================================
# Part 1: Partitioning & Reordering (算法核心)
# ==============================================================================

def partition_hybrid_manifold(num_nodes, edge_index, edge_ts, hot_mask, num_parts, part_type = 'louvain', num_micro_parts = 128):#metis):
    """
    Step 1: 基于时序流形的图分区算法。
    将图划分为 num_parts 个分区，返回每个节点所属的分区 ID 和微聚类 ID。
    """

        
    print("[Step 1] Running Hybrid Manifold Partitioning...")
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    ts = edge_ts.cpu().numpy()

    # # 1.1 残差图构建 (仅针对冷节点)
    # hot_mask_np = hot_mask.cpu().numpy()
    # cold_edges_mask = (~hot_mask_np[src]) & (~hot_mask_np[dst])
    # cold_src = src[cold_edges_mask]
    # cold_dst = dst[cold_edges_mask]
    
    # 1.2 Louvain 聚类
    
    if part_type == 'louvain':
        if len(ts) > 0:
            nx_graph = nx.Graph()
            nx_graph.add_nodes_from(np.arange(num_nodes))
             #nx_graph.add_edges_from(zip(cold_src, cold_dst))
            nx_graph.add_edges_from(zip(src,dst))
            print(f"    - Micro-clustering on residual graph ({nx_graph.number_of_nodes()} nodes)...")
            partition_map = community_louvain.best_partition(nx_graph, resolution=3.0)
            #_, membership = pymetis.part_graph(num_micro_parts, adjacency=adj_list)
            #node_to_cluster = np.array(membership, dtype=np.int32)
        else:
            partition_map = {}
        
        node_to_cluster = np.full(num_nodes, -1, dtype=np.int32)
        max_cluster_id = 0
        if partition_map:
            nodes = np.array(list(partition_map.keys()))
            clusters = np.array(list(partition_map.values()))
            node_to_cluster[nodes] = clusters
            max_cluster_id = clusters.max() + 1
    elif part_type == 'meits':
        adj_list = [[] for _ in range(num_nodes)]
        for s, d in zip(src, dst):
            adj_list[s].append(d)
            adj_list[d].append(s)
        _, membership = pymetis.part_graph(num_micro_parts, adjacency=adj_list)
        node_to_cluster = np.array(membership, dtype=np.int32)
        max_cluster_id = node_to_cluster.max() + 1
    # 处理孤立点
    cold_indices = np.arange(num_nodes)#np.where(~hot_mask_np)[0]
    unclustered_mask = node_to_cluster[cold_indices] == -1
    isolated_nodes = cold_indices[unclustered_mask]
    if len(isolated_nodes) > 0:
        new_ids = np.arange(max_cluster_id, max_cluster_id + len(isolated_nodes))
        node_to_cluster[isolated_nodes] = new_ids
        max_cluster_id += len(isolated_nodes)
    
    src_cids = node_to_cluster[src]
    # 获取 dst 节点的 cluster
    dst_cids = node_to_cluster[dst]
    
    # 拼接：所有“点-边时间”对
    combined_cids = np.concatenate([src_cids, dst_cids])
    combined_ts = np.concatenate([ts, ts])
    
    # 过滤掉 -1 (未聚类点)
    mask_c = combined_cids != -1
    final_c = combined_cids[mask_c]
    final_t = combined_ts[mask_c]
    
    # 排序与分组统计 (逻辑不变)
    sort_idx = np.lexsort((final_t, final_c))
    sorted_c = final_c[sort_idx]
    sorted_t = final_t[sort_idx]
    
    unique_c, split_idx = np.unique(sorted_c, return_index=True)
    cluster_ts_groups = np.split(sorted_t, split_idx[1:])
    
    cluster_feats = np.zeros((len(unique_c), 2)) # Mean, Std
    for i, t_list in enumerate(cluster_ts_groups):
        if len(t_list) > 0:
            cluster_feats[i, 0] = np.mean(t_list)
            cluster_feats[i, 1] = np.std(t_list)
            
    # 1.4 分配 (逻辑不变)
    # 按时序特征对 Cluster 进行排序并分配 Partition
    sort_keys = np.lexsort((cluster_feats[:, 1], cluster_feats[:, 0]))
    sorted_cids = unique_c[sort_keys]
    
    cluster_to_part = {}
    block_num = num_parts * 64
    block_size = (max_cluster_id + 1) // block_num + 1
    for i, cid in enumerate(sorted_cids):
        part_id = (i // block_size) % num_parts
        cluster_to_part[cid] = part_id
        
    final_parts = torch.zeros(num_nodes, dtype=torch.int64)
    lookup = np.full(max_cluster_id + 1, 0, dtype=np.int32) # 默认 0
    if len(cluster_to_part) > 0:
        c_ids = np.array(list(cluster_to_part.keys()))
        p_ids = np.array(list(cluster_to_part.values()))
        lookup[c_ids] = p_ids
        
    mask_has_cluster = node_to_cluster != -1
    final_parts[mask_has_cluster] = torch.from_numpy(lookup[node_to_cluster[mask_has_cluster]]).long()
    print("     - cluster distribution:", torch.from_numpy(node_to_cluster).unique(return_counts=True))
    print(f"    - Final partition distribution: {final_parts.unique(return_counts=True)}")
    print(f"    - Total clusters: {len(cluster_to_part)} (Max Cluster ID: {max_cluster_id})(max )")
    return final_parts, node_to_cluster

    # # 1.3 时序特征提取与排序
    # print("    - Extracting temporal features for clusters...")
    # #mask_valid = ~hot_mask_np[src]
    # valid_src = src#[mask_valid]
    # valid_ts = ts#[mask_valid]
    # edge_cids = node_to_cluster[valid_src]#[valid_src]
    
    # mask_c = edge_cids != -1
    # final_c = edge_cids[mask_c]
    # final_t = valid_ts[mask_c]
    
    # # 按 (Cluster, Time) 排序
    # sort_idx = np.lexsort((final_t, final_c))
    # sorted_c = final_c[sort_idx]
    # sorted_t = final_t[sort_idx]
    
    # unique_c, split_idx = np.unique(sorted_c, return_index=True)
    # cluster_ts_groups = np.split(sorted_t, split_idx[1:])
    
    # cluster_feats = np.zeros((len(unique_c), 2)) # Mean, Std
    # for i, t_list in enumerate(cluster_ts_groups):
    #     if len(t_list) > 0:
    #         cluster_feats[i, 0] = np.mean(t_list)
    #         cluster_feats[i, 1] = np.std(t_list)
            
    # # 1.4 分配
    # # 按 std 和 mean 排序，将相似时序模式的 cluster 分在一起
    # sort_keys = np.lexsort((cluster_feats[:, 1], cluster_feats[:, 0]))
    # sorted_cids = unique_c[sort_keys]
    
    # cluster_to_part = {}
    # for i, cid in enumerate(sorted_cids):
    #     part_id = (i // block_size) % num_parts
    #     cluster_to_part[cid] = part_id
        
    # final_parts = torch.zeros(num_nodes, dtype=torch.int64)
    # lookup = np.full(max_cluster_id + 1, -1, dtype=np.int32)
    # if len(cluster_to_part) > 0:
    #     c_ids = np.array(list(cluster_to_part.keys()))
    #     p_ids = np.array(list(cluster_to_part.values()))
    #     lookup[c_ids] = p_ids
        
    # mask_has_cluster = node_to_cluster != -1
    # final_parts[mask_has_cluster] = torch.from_numpy(lookup[node_to_cluster[mask_has_cluster]]).long()
    # #final_parts[final_parts == -1] = torch.randint(0, num_parts, ((final_parts == -1).sum().item(),))
    # # 热点随机分配 (负载均衡)
    # #if hot_mask.sum() > 0:
    # #    final_parts[hot_mask] = torch.randint(0, num_parts, (hot_mask.sum().item(),))
    
    # print(final_parts.unique(return_counts=True))
    # return final_parts, node_to_cluster

def hierarchical_spatiotemporal_reordering(graph, node_parts, hot_mask, node_avg_ts, num_parts, num_time_buckets=8):
    """
    Step 2: 生成全局 NewID 的重排映射 (Permutation)。
    """
    print(f"\n[Step 2] Calculating Reordering Permutation...")
    final_layout = []
    
    node_parts_np = node_parts.cpu().numpy()
    hot_mask_np = hot_mask.cpu().numpy()
    node_avg_ts_np = node_avg_ts.cpu().numpy()
    
    try:
        full_adj = graph.adj_external(scipy_fmt='csr')
    except AttributeError:
    # 如果是非常老版本的 DGL
        full_adj = graph.adj(scipy_fmt='csr')
    
    for pid in tqdm(range(num_parts), desc="Reordering Partitions"):
        p_nodes = np.where(node_parts_np == pid)[0]
        if len(p_nodes) == 0: continue
        
        is_hot = hot_mask_np[p_nodes]
        local_hubs = p_nodes[is_hot]
        local_cold = p_nodes[~is_hot]
        
        # Layer 1: Hubs (按度数)
        if len(local_hubs) > 0:
            degs = graph.in_degrees(torch.from_numpy(local_hubs)).numpy()
            local_hubs = local_hubs[np.argsort(-degs)]
        
        # Layer 2: Cold (Halo-Augmented RCM)
        cold_layout = []
        if len(local_cold) > 0:
            # 按时间分桶
            times = node_avg_ts_np[local_cold]
            buckets = np.array_split(local_cold[np.argsort(times)], num_time_buckets)
            
            for bucket in buckets:
                if len(bucket) == 0: continue
                # 提取桶内子图 + Halo
                sub_csr = full_adj[bucket, :]
                neighbors = np.unique(sub_csr.indices)
                halo = np.setdiff1d(neighbors, bucket)
                
                # 构建增强邻接矩阵进行 RCM
                all_nodes = np.concatenate([bucket, halo])
                sorter = np.argsort(all_nodes)
                sorted_all = all_nodes[sorter]
                
                # 映射 indices 到 local 0..N
                mapped_indices = sorter[np.searchsorted(sorted_all, sub_csr.indices)]
                
                # 构造 CSR
                data = np.ones(len(mapped_indices), dtype=np.int8)
                # Halo 节点的行是空的 (indptr 不变)
                new_indptr = np.concatenate([sub_csr.indptr, np.full(len(halo), sub_csr.indptr[-1])])
                
                aug_mat = sp.sparse.csr_matrix((data, mapped_indices, new_indptr), shape=(len(all_nodes), len(all_nodes)))
                
                perm = reverse_cuthill_mckee(aug_mat, symmetric_mode=True)
                # 只保留 bucket 内的节点
                local_perm = perm[perm < len(bucket)]
                cold_layout.append(bucket[local_perm])
                
        local_cold_sorted = np.concatenate(cold_layout) if cold_layout else np.array([], dtype=np.int64)
        final_layout.append(np.concatenate([local_hubs, local_cold_sorted]))
        
    full_perm = np.concatenate(final_layout)
    
    # 补全可能遗漏的节点 (通常是孤立且未被分配的)
    if len(full_perm) != graph.num_nodes():
        mask = np.ones(graph.num_nodes(), dtype=bool)
        mask[full_perm] = False
        missing = np.where(mask)[0]
        if len(missing) > 0:
            full_perm = np.concatenate([full_perm, missing])
            
    return torch.from_numpy(full_perm)

# ==========================================
# Part 2: Chunk Generation
# ==========================================

def prepare_spatiotemporal_chunks(
    edge_index: torch.Tensor, edge_ts: torch.Tensor, 
    node_parts: torch.Tensor, node_clusters: np.ndarray,
    num_parts: int, slice_param: Tuple[str, int], 
    max_events_per_chunk: int, output_dir: Path
):
    """
    Step 3: 生成 Chunks (Slot)。输入必须已经是 NewID。
    """
    print(f"\n[Step 3] Generating Spatio-Temporal Chunks...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 全局时间切分
    if slice_param[0] == "event":
        # 滚动边界逻辑 (简化版)
        u_ts, counts = torch.unique(edge_ts, return_counts=True)
        cum = torch.cumsum(counts, 0)
        boundaries = [edge_ts.min().item()]
        curr = 0
        total = cum[-1].item()
        while curr < total:
            target = curr + slice_param[1]
            if target >= total: break
            idx = torch.searchsorted(cum, target)
            boundaries.append(u_ts[idx].item())
            curr = cum[idx].item()
        if boundaries[-1] < edge_ts.max().item(): boundaries.append(edge_ts.max().item() + 1)
        time_boundaries = torch.tensor(boundaries)
    else:
        time_boundaries = torch.linspace(edge_ts.min(), edge_ts.max() + 1, slice_param[1])
        
    torch.save({"boundaries": time_boundaries, "strategy": slice_param}, output_dir / "dist_meta.pt")
    
    src, dst = edge_index
    # 边的 Owner 由 Dst 决定 (用于 In-Edge aggregation)
    edge_owners = node_parts[dst]
    node_clusters_t = torch.from_numpy(node_clusters).to(edge_index.device).long()
    eids = torch.arange(len(src), device=src.device)
    
    for pid in tqdm(range(num_parts)):
        part_dir = output_dir / f"part_{pid}"
        part_dir.mkdir(exist_ok=True)
        
        mask = (edge_owners == pid)
        p_src, p_dst, p_ts, p_eid = src[mask], dst[mask], edge_ts[mask], eids[mask]
        
        # 按时间排序
        sort_idx = torch.argsort(p_ts)
        p_src, p_dst, p_ts, p_eid = p_src[sort_idx], p_dst[sort_idx], p_ts[sort_idx], p_eid[sort_idx]
        
        slot_indices = torch.searchsorted(p_ts, time_boundaries)
        global_ctr = 0
        
        ctrlen = []
        
        for tid in tqdm(range(len(time_boundaries)-1), desc=f"Part {pid}"):
            s, e = slot_indices[tid], slot_indices[tid+1]
            if s >= e: continue
            
            # 1. 提取当前 Slot 数据
            sl_src, sl_dst, sl_ts, sl_eid = p_src[s:e], p_dst[s:e], p_ts[s:e], p_eid[s:e]
            
            # 2. 核心排序：Cluster ID 为第一关键字，时间为第二关键字
            # 这保证了"相似"(ID相近)的 Cluster 在物理内存上是连续的
            sl_cids = node_clusters_t[sl_dst]
            # 处理未聚类节点 (-1)
            sl_cids[sl_cids == -1] = 999999
            
            sort_k = np.lexsort((sl_ts.cpu().numpy(), sl_cids.cpu().numpy()))
            sort_k_t = torch.from_numpy(sort_k)
            
            sl_src, sl_dst, sl_ts, sl_eid, sl_cids = \
                sl_src[sort_k_t], sl_dst[sort_k_t], sl_ts[sort_k_t], sl_eid[sort_k_t], sl_cids[sort_k_t]
            
            # --- [Modification Start] 基于累积分布的动态均衡切分 ---
            num_items = len(sl_src)
            if num_items > 0:
                # 3. 确定切分份数 (K)
                # max_events_per_chunk 仅作为一个"量级参考"，用于决定切几份，不再作为硬性上限
                # 如果总数很少，至少切 1 份
                num_subs = max(1, int(np.round(num_items / max_events_per_chunk)))
                target_size = num_items / num_subs
                
                # 4. 构建 Cluster 粒度的累积分布
                # unique_consecutive: 获取每个连续 Cluster 的大小
                # 注意：因为已经按 CID 排序，这里得到的 counts 就是每个 Cluster (及其所有时间步边) 的总数
                _, counts = torch.unique_consecutive(sl_cids, return_counts=True)
                
                # cumsum: [c1, c1+c2, c1+c2+c3, ...] -> 潜在的完美切分点
                cluster_boundaries = torch.cumsum(counts, dim=0).cpu().numpy()
                
                start_idx = 0
                
                for k in range(num_subs):
                    # 最后一包直接收尾，防止精度误差丢数据
                    if k == num_subs - 1:
                        end_idx = num_items
                    else:
                        # 5. 寻找最佳切分点
                        # 理想切分位置：当前应该是第 (k+1) 份的结束
                        ideal_boundary = (k + 1) * target_size
                        
                        # 在 cluster_boundaries 中搜索最接近 ideal_boundary 的位置
                        # searchsorted 返回插入位置，使得 left <= ideal < right
                        idx = np.searchsorted(cluster_boundaries, ideal_boundary)
                        
                        # 获取"前一个边界"和"后一个边界"作为候选
                        # 前候选项 (idx-1)
                        cand_prev = cluster_boundaries[idx-1] if idx > 0 else 0
                        # 后候选项 (idx)
                        cand_next = cluster_boundaries[idx] if idx < len(cluster_boundaries) else num_items
                        
                        # 6. 择优录取 (综合考虑前后信息)
                        # 比较哪个边界离理想值更近，从而使当前 chunk 和下一个 chunk 的负载更均衡
                        dist_prev = abs(cand_prev - ideal_boundary)
                        dist_next = abs(cand_next - ideal_boundary)
                        
                        if dist_prev <= dist_next:
                            best_cut = cand_prev
                        else:
                            best_cut = cand_next
                        
                        # 确保进度向前 (不要切出空包)
                        end_idx = max(start_idx + 1, int(best_cut))
                        # 确保不越界
                        end_idx = min(end_idx, num_items)

                    # 7. 保存 Sub-chunk
                    # 只有当非空时才保存
                    if end_idx > start_idx:
                        ctrlen.append(end_idx - start_idx)
                        chunk = {
                            "src": sl_src[start_idx:end_idx], 
                            "dst": sl_dst[start_idx:end_idx], 
                            "ts": sl_ts[start_idx:end_idx], 
                            "eid": sl_eid[start_idx:end_idx],
                            "cid": torch.unique(sl_cids[start_idx:end_idx]),
                            "slot_id": tid, 
                            "chunk_id": global_ctr
                        }
                        torch.save(chunk, part_dir / f"slot_{tid:04d}_sub_{k:04d}.pt")
                        global_ctr += 1
                        
                        # 更新下一轮起点
                        start_idx = end_idx
            print(f"- Slot {tid}: {num_items} edges split into {num_subs} chunks.Min{min(ctrlen)}, Max{max(ctrlen)}, Avg{np.mean(ctrlen):.2f}")

        
        # for tid in tqdm(range(len(time_boundaries)-1), desc=f"Part {pid}"):
        #     s, e = slot_indices[tid], slot_indices[tid+1]
        #     if s >= e: continue
            
        #     sl_src, sl_dst, sl_ts, sl_eid = p_src[s:e], p_dst[s:e], p_ts[s:e], p_eid[s:e]
            
        #     # 微批次 Cluster 排序
        #     sl_cids = node_clusters_t[sl_dst]
        #     sl_cids[sl_cids == -1] = 999999
            
        #     sort_k = np.lexsort((sl_ts.cpu().numpy(), sl_cids.cpu().numpy()))
        #     sort_k_t = torch.from_numpy(sort_k)
            
        #     cpsl_src, cpsl_dst, cpsl_ts, cpsl_eid, cpsl_cids = sl_src[sort_k_t], sl_dst[sort_k_t], sl_ts[sort_k_t], sl_eid[sort_k_t], sl_cids[sort_k_t]
            
        #     sl_src, sl_dst, sl_ts, sl_eid, sl_cids = cpsl_src, cpsl_dst, cpsl_ts, cpsl_eid, cpsl_cids
            
        #     # sl_src, sl_dst, sl_ts, sl_eid, sl_cids = \
        #     #     sl_src[sort_k_t], sl_dst[sort_k_t], sl_ts[sort_k_t], sl_eid[sort_k_t], sl_cids[sort_k_t]
            
        #     # 切分 Sub-chunks
        #     num_items = len(sl_src)
        #     num_subs = (num_items + max_events_per_chunk - 1) // max_events_per_chunk
            
        #     for k in range(num_subs):
        #         cs, ce = k*max_events_per_chunk, min((k+1)*max_events_per_chunk, num_items)
        #         chunk = {
        #             "src": sl_src[cs:ce], "dst": sl_dst[cs:ce], 
        #             "ts": sl_ts[cs:ce], "eid": sl_eid[cs:ce],
        #             "cid": torch.unique(sl_cids[cs:ce]),
        #             "slot_id": tid, "chunk_id": global_ctr
        #         }
        #         torch.save(chunk, part_dir / f"slot_{tid:04d}_sub_{k:04d}.pt")
        #         global_ctr += 1

# ==============================================================================
# Part 3: Distributed Metadata & Feature Storage (重点修改)
# ==============================================================================

def prepare_distributed_metadata(node_parts, edge_index, num_parts, output_dir):
    """
    Step 4: 生成 Partition Book 和 Replica Table。
    """
    print(f"\n[Step 4] Generating Distributed Metadata...")
    src, dst = edge_index
    edge_parts = node_parts[dst] # NewID Base
    partition_book = []
    
    for pid in tqdm(range(num_parts), desc="Partition Book"):
        # Owned Nodes
        owned = torch.nonzero(node_parts == pid, as_tuple=True)[0]
        
        # Halo Nodes: dst 在 pid, 但 src 不在 pid
        mask_p = (edge_parts == pid)
        p_src = src[mask_p]
        src_owners = node_parts[p_src]
        mask_halo = (src_owners != pid)
        halos = torch.unique(p_src[mask_halo])
        
        # 存储顺序: [Owned, Halo]
        # 记录 owned 的长度，方便加载时区分
        partition_book.append(torch.cat([owned, halos]))
    
    torch.save((partition_book,node_parts, edge_parts), output_dir / "partition_book.pt")
    # Replica Table
    rep_table = build_replica_table(len(node_parts), partition_book, num_parts)
    torch.save(rep_table, output_dir / "replica_table.pt")
    
    return partition_book, rep_table

def save_distributed_context(
    output_dir: Path,
    num_parts: int,
    partition_book: List[torch.Tensor],
    edge_owner_part: torch.Tensor, # 用于切分 Edge Feature
    node_feat: Union[torch.Tensor, List[torch.Tensor], None] = None,
    node_label: Union[torch.Tensor, List[torch.Tensor], None] = None,
    edge_feat: Union[torch.Tensor, List[torch.Tensor], None] = None,
    edge_label: Union[torch.Tensor, List[torch.Tensor], None] = None,
):
    """
    Step 5: 将所有特征（点/边、特征/标签）按分区切分并打包存储。
    支持 Tensor 或 List[Tensor] (用于离散时间特征)。
    输入数据必须已经按 NewID (Node) 或 New Edge Order (Edge) 排序。
    """
    print(f"\n[Step 5] Saving Distributed Features & Labels...")
    
    def _slice_data(data, indices):
        """辅助函数：对 Tensor 或 List[Tensor] 进行切片"""
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data[indices]
        if isinstance(data, list):
            # 针对每个时间步的 Tensor 进行切片
            return [t[indices] for t in data]
        return None

    # 预先计算 Edge Mask 列表以节省内存
    edge_masks = []
    if edge_feat is not None or edge_label is not None:
        for pid in range(num_parts):
            edge_masks.append(edge_owner_part == pid)

    for pid in tqdm(range(num_parts), desc="Saving Context"):
        part_dir = output_dir / f"part_{pid}"
        part_dir.mkdir(exist_ok=True)
        
        context_data = {}
        
        # 1. 存储节点相关数据 (基于 Partition Book: Owned + Halo)
        # 只要是 Partition Book 里的节点，本地都需要有特征副本
        nodes = partition_book[pid]
        
        if node_feat is not None:
            context_data['node_feat'] = _slice_data(node_feat, nodes)
            
        if node_label is not None:
            # 注意：Halo 节点的标签通常不需要（训练只针对 Owned），
            # 但为了统一格式，这里还是全存，加载时只取 Owned 即可。
            context_data['node_label'] = _slice_data(node_label, nodes)
            
        # 2. 存储边相关数据 (基于 Edge Owner: Only Owned Edges)
        # 边的特征只存储在负责该边的分区上 (Edge Owner)
        if edge_feat is not None or edge_label is not None:
            mask = edge_masks[pid]
            
            if edge_feat is not None:
                context_data['edge_feat'] = _slice_data(edge_feat, mask)
                
            if edge_label is not None:
                context_data['edge_label'] = _slice_data(edge_label, mask)
        
        # 保存到单个文件
        if context_data:
            torch.save(context_data, part_dir / "distributed_context.pt")

# ==============================================================================
# Part 4: Route Pre-computation
# ==============================================================================

def precompute_slot_routes(output_dir, num_parts, rep_table, partition_book, node_parts):
    """
    Step 6: 计算通信路由。
    """
    print(f"\n[Step 6] Pre-computing Routes...")
    
    # 构建 Local Index Map (Global NewID -> Index in PartitionBook)
    # 用于告诉发送方：你要发的那个节点，在你本地 PartitionBook 的哪个位置
    my_maps = []
    for nodes in partition_book:
        # 使用 Tensor 做映射 (假设内存够)
        m = torch.full((len(node_parts),), -1, dtype=torch.long)
        m[nodes] = torch.arange(len(nodes), dtype=torch.long)
        my_maps.append(m)
        
    for pid in range(num_parts):
        part_dir = output_dir / f"part_{pid}"
        local_map = my_maps[pid]
        
        for p_slot in tqdm(list(part_dir.glob("slot_*.pt")), desc=f"Part {pid}"):
            chunk = torch.load(p_slot)
            
            # 找出 chunk 中所有节点
            nodes = torch.cat([chunk['src'], chunk['dst']])
            u_nodes = torch.unique(nodes)
            
            # 筛选：只有我自己拥有的节点，我才需要发送给副本
            owners = node_parts[u_nodes]
            my_nodes = u_nodes[owners == pid]
            
            if len(my_nodes) == 0:
                chunk['route'] = None # 或者 Empty Route
                torch.save(chunk, p_slot)
                continue
                
            # 查表：谁需要这些节点？
            q_idx, q_ranks, q_remote_locs = rep_table.lookup(my_nodes)
            
            if len(q_ranks) > 0:
                s_gids = my_nodes[q_idx]
                s_local = local_map[s_gids] # 转为本地索引
                
                # 按 rank 排序打包
                sort_idx = torch.argsort(q_ranks)
                final_ranks = q_ranks[sort_idx]
                final_local = s_local[sort_idx]
                final_remote = q_remote_locs[sort_idx]
                
                # 计算每个 Rank 的发送量
                u_ranks, counts = torch.unique(final_ranks, return_counts=True)
                send_sizes = torch.zeros(num_parts, dtype=torch.long)
                send_sizes[u_ranks.long()] = counts.long()
                
                route = CommPlan(
                    send_ranks=final_ranks,
                    send_sizes=send_sizes,
                    send_indices=final_local,
                    send_remote_indices=final_remote
                )
            else:
                route = None
                
            chunk['route'] = route
            torch.save(chunk, p_slot)

# ==============================================================================
# Part 5: Main Execution
# ==============================================================================

if __name__ == "__main__":
    # 配置路径
    src_root = Path("/mnt/data/zlj/starrygl-data/ctdg").resolve()
    tgt_root = Path("/mnt/data/zlj/starrygl-data/nparts").resolve()
    num_parts = 4
    hot_ratio = 0.1
    tgt_root.mkdir(parents=True, exist_ok=True)
    
    # 模拟工具函数
    def get_temporal_stats(num_nodes, u, v, t):
        # 简单实现，替换为你之前的 get_node_temporal_stats
        return torch.zeros(num_nodes)

    for p_path in src_root.glob("*.pth"):
        name = p_path.stem
        if name != "StackOverflow" and name != 'WikiTalk': continue # 调试用
        
        print(f"=== Processing {name} ===")
        data = torch.load(p_path)
        num_nodes = data['num_nodes']
        ds = data['dataset']
        
        # 1. 提取全图拓扑与时间
        # 兼容不同格式
        if isinstance(ds, dict):
            # 单个 tensor
            edge_index = ds['edge_index']
            edge_ts = ds.get('edge_ts')
            if edge_ts is None and edge_index.shape[0] > 2:
                edge_ts = edge_index[2]
                edge_index = edge_index[:2]
            
            # 提取特征 (Tensor 或 List[Tensor])
            node_feat = ds.get('node_feat') 
            node_label = ds.get('y') if 'y' in ds else None
            edge_feat = ds.get('edge_feat')
            edge_label = ds.get('edge_label') if 'edge_label' in ds else None
        else:
            # List[Data] 格式合并
            edge_index = torch.cat([d['edge_index'] for d in ds], dim=1)
            edge_ts = torch.cat([torch.full((d['edge_index'].shape[1],), i) for i, d in enumerate(ds)])
            # 特征合并逻辑需视情况而定，这里略过复杂合并，假设已预处理好
            node_feat = [d['x'] for d in ds] if 'x' in ds[0] else None#换成TensorData吧
            node_label = [d['y'] for d in ds] if 'y' in ds[0] else None #换成TensorData吧
            edge_feat = torch.cat([d['edge_feat'] for d in ds], dim=1) if 'edge_feat' in ds[0] else None
            edge_label = torch.cat([d['edge_label'] for d in ds], dim=1) if 'edge_label' in ds[0] else None

        # 2. 识别热点
        is_hot = torch.zeros(num_nodes, dtype=torch.bool)
        if hot_ratio > 0:
            deg = torch.bincount(edge_index.flatten(), minlength=num_nodes)
            val, idx = torch.topk(deg, int(num_nodes * hot_ratio))
            is_hot[idx] = True

        # 3. 分区 (基于 OldID)
        parts, clusters = partition_hybrid_manifold(num_nodes, edge_index, edge_ts, is_hot, num_parts)
        
        # 4. 重排 (生成 Permutation: Old -> New)
        # 需要计算节点平均时间
        avg_ts = get_temporal_stats(num_nodes, edge_index[0], edge_index[1], edge_ts)
        # 构造临时图用于计算度
        g_tmp = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
        
        perm = hierarchical_spatiotemporal_reordering(g_tmp, parts, is_hot, avg_ts, num_parts)
        
        # 5. [CRITICAL] 应用 Permutation 到所有数据
        print("[Step Main] Applying Permutation (Mapping to NewID)...")
        # 建立 Old -> New 映射
        # rev_perm[old_id] = new_id
        rev_perm = torch.empty_like(perm)
        rev_perm[perm] = torch.arange(num_nodes, device=perm.device)
        # [修改后的代码]
        save_dir = tgt_root / f"{name}_{num_parts:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)  # <--- 关键：先创建目录
        torch.save((perm, rev_perm), save_dir / "perm.pt")
        # 5.1 映射拓扑
        new_edge_index = torch.stack([rev_perm[edge_index[0]], rev_perm[edge_index[1]]])
        new_parts = parts[perm] # parts 数组重排
        new_clusters = clusters[perm.cpu().numpy()]
        print(parts, new_parts)
        # 5.2 映射点特征/标签 (支持 List)
        new_node_feat = None
        if node_feat is not None:
            if isinstance(node_feat, list):
                new_node_feat = [nf[perm] for nf in node_feat]
            else:
                new_node_feat = node_feat[perm]
                
        new_node_label = None
        if node_label is not None:
            if isinstance(node_label, list):
                new_node_label = [nl[perm] for nl in node_label]
            else:
                new_node_label = node_label[perm]
        
        # 5.3 映射边特征/标签
        # 注意：重排节点不改变边的顺序，除非我们也对边进行了重排
        # 但在 prepare_spatiotemporal_chunks 里，我们是对边进行了时间排序的
        # 为了让 edge_feat 能对齐 chunk 里的 eid，我们这里暂不改变边的顺序
        # 边的顺序将在 Chunking 阶段被改变，但 Chunking 只保存 eid
        # **关键点**：save_distributed_context 需要原始的边数据对应到边所属的 Partition
        # 我们这里的 edge_feat 依然对应 edge_index 的顺序
        
        output_dir = tgt_root / f"{name}_{num_parts:03d}"
        
        # 6. 生成 Chunks
        prepare_spatiotemporal_chunks(
            new_edge_index, edge_ts, 
            new_parts, new_clusters, 
            num_parts, ("event", 4000), 200, 
            output_dir
        )
        
        # 7. 生成元数据
        p_book, rep_table = prepare_distributed_metadata(new_parts, new_edge_index, num_parts, output_dir)
        
        # 8. 保存分布式特征 (新增功能)
        # 计算边的 Owner (NewID Based)
        edge_owner_part = new_parts[new_edge_index[1]]
        
        save_distributed_context(
            output_dir, num_parts, p_book, edge_owner_part,
            node_feat=new_node_feat,
            node_label=new_node_label,
            edge_feat=edge_feat,   # 原始顺序，但在函数内会通过 mask 切分
            edge_label=edge_label
        )
        
        
        # 10. 清理内存
        del perm, rev_perm, new_edge_index, new_node_feat
        gc.collect()
        
        print(f"Done {name}.")