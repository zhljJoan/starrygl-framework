import os
import sys
import argparse
import torch
import numpy as np
import random
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt # 可选：用于画分布图

# === 路径注入 (根据你的项目结构调整) ===
current_file = Path(__file__).resolve()
project_root = current_file.parent
sys.path.append(str(project_root))

# 尝试引入必要组件，如果失败则尝试仅使用 Torch
try:
    from starrygl.data.batch import AtomicBatch
    from starrygl.cache.route import CommPlan
except ImportError:
    print("[Warning] StarryGL modules not found. Using raw torch.load (might fail on custom classes).")

class PreprocessingEvaluator:
    def __init__(self, data_root, dataset, num_parts):
        self.root = Path(data_root)
        self.dataset = dataset
        self.num_parts = num_parts
        self.suffix = f"{dataset}_{num_parts:03d}"
        
        # 路径定义
        self.nparts_dir = self.root / "nparts" 
        self.processed_dir_base = self.root / "processed_atomic" / self.suffix
        
        # 查找真实的 nparts 目录 (处理可能的后缀)
        candidates = list(self.nparts_dir.glob(f"{self.suffix}*"))
        if not candidates:
            raise FileNotFoundError(f"Cannot find nparts directory for {self.suffix}")
        self.meta_dir = candidates[0]
        
        print(f"✅ Target Metadata: {self.meta_dir}")
        print(f"✅ Target Chunks: {self.processed_dir_base}")
        print("-" * 60)

    def load_partition_book(self):
        """加载 Partition Book，兼容多种存储格式"""
        pb_path = self.meta_dir / "partition_book.pt"
        data = torch.load(pb_path, map_location='cpu')
        
        # 兼容 tuple (book, local_ids...) 或 list 或 直接 tensor list
        if isinstance(data, (tuple, list)):
            return data[0]
        return data

    def eval_load_balance(self):
        """
        1. 负载均衡评估
        - Computation Balance: Owned Nodes 分布
        - Memory Balance: Total Stored (Owned + Halo) 分布
        """
        print("\n📊 [Metric 1] Load Balancing Analysis")
        p_book = self.load_partition_book()
        
        owned_counts = []
        stored_counts = []
        
        for rank in range(self.num_parts):
            # 1. Owned Nodes
            n_owned = len(p_book[rank])
            owned_counts.append(n_owned)
            
            # 2. Stored Nodes (从 distributed_context 读取)
            ctx_path = self.meta_dir / f"part_{rank}" / "distributed_context.pt"
            if ctx_path.exists():
                # 只读 Metadata，不加载大 Tensor
                # 这种 trick 依赖于 pytorch 版本，如果是 zip 格式通常需要加载
                # 这里我们完整加载但立即释放
                ctx = torch.load(ctx_path, map_location='cpu')
                if 'node_feat' in ctx:
                    n_stored = ctx['node_feat'].shape[0]
                else:
                    # 如果没有 feature，假设 stored = map 长度
                    # 这里假设 context 里有 local_map
                    n_stored = n_owned # Fallback
                stored_counts.append(n_stored)
            else:
                print(f"  [Warn] Context missing for rank {rank}")
                stored_counts.append(n_owned)

        owned = np.array(owned_counts)
        stored = np.array(stored_counts)
        
        # 计算指标
        print(f"  > Computation Load (Owned Nodes):")
        print(f"    - Mean: {owned.mean():.1f}")
        print(f"    - Std Dev: {owned.std():.1f}")
        print(f"    - CV (Coeff of Variation): {owned.std()/owned.mean():.4f} (Ideal: 0.0)")
        
        print(f"  > Memory Load (Stored Nodes):")
        print(f"    - Mean: {stored.mean():.1f}")
        print(f"    - Max/Mean Ratio: {stored.max()/stored.mean():.4f} (Ideal: 1.0)")
        
        return stored.sum() # 返回总物理存储量供下一步使用

    def eval_communication(self, total_stored_nodes):
        """
        2. 通信开销评估
        - Replication Factor (RF)
        - Halo Ratio
        """
        print("\n📡 [Metric 2] Communication Cost")
        
        p_book = self.load_partition_book()
        # 近似全局节点数 (假设 partition 覆盖全图且互斥)
        total_unique_nodes = sum([len(b) for b in p_book])
        
        # Replication Factor = 总存储节点数 / 实际唯一节点数
        rf = total_stored_nodes / max(1, total_unique_nodes)
        
        # Halo Ratio = Halo / Stored
        total_halo = total_stored_nodes - total_unique_nodes
        halo_ratio = total_halo / max(1, total_stored_nodes)
        
        print(f"  - Total Unique Nodes: {total_unique_nodes}")
        print(f"  - Total Physical Nodes (Sum over ranks): {total_stored_nodes}")
        print(f"  > Replication Factor (RF): {rf:.4f}")
        print(f"    (Interpretation: Each node is stored on {rf:.2f} GPUs on average)")
        print(f"  > Avg Halo Ratio: {halo_ratio:.2%}")
        
        if rf > 2.0:
            print("    ⚠️ [Alert] High RF detected! Check partitioning algorithm.")
        else:
            print("    ✅ [Pass] RF is within acceptable range (< 2.0).")

    def eval_temporal_integrity(self, sample_ratio=0.1):
        """
        3. 时序一致性检查
        - 检查 Batch 间的时间单调性
        - 检查 Batch 内的信息泄露 (Edge TS > Batch TS)
        """
        print("\n⏳ [Metric 3] Temporal Integrity Check")
        
        # 扫描任一分区的 chunk 文件
        chunk_dir = self.processed_dir_base / "part_0"
        if not chunk_dir.exists():
            print(f"  [Error] Chunk dir not found: {chunk_dir}")
            return

        files = sorted(list(chunk_dir.glob("slot_*.pt")), key=lambda x: x.name)
        num_files = len(files)
        if num_files == 0:
            print("  [Error] No slot files found.")
            return
            
        # 抽样
        indices = sorted(random.sample(range(num_files), max(1, int(num_files * sample_ratio))))
        print(f"  - Scanning {len(indices)}/{num_files} files for violations...")
        
        violations = 0
        leakages = 0
        prev_max_ts = -1.0
        
        for idx in indices:
            f = files[idx]
            try:
                # 加载 AtomicBatch
                raw = torch.load(f, map_location='cpu')
                # 兼容格式: 可能是 AtomicBatch 对象，也可能是 list
                if hasattr(raw, 'task_data'):
                    task = raw.task_data
                    layers = raw.layer_data
                elif isinstance(raw, list):
                    task = raw[0]
                    layers = raw[1:]
                else:
                    continue

                # 1. 检查 Batch 时间范围
                # 假设 task['ts'] 或 task['task_ts'] 存在
                ts_key = 'task_ts' if 'task_ts' in task else 'ts'
                if ts_key not in task:
                    # 可能是 task_start / task_end
                    current_min = task.get('time_start', 0)
                    current_max = task.get('time_end', 0)
                else:
                    ts_tensor = task[ts_key]
                    current_min = ts_tensor.min().item()
                    current_max = ts_tensor.max().item()
                
                if current_min < prev_max_ts:
                    violations += 1
                prev_max_ts = current_max
                
                # 2. 检查子图泄露 (Leakage)
                # 任何子图边的生成时间不得晚于 Task 的发生时间
                for layer in layers:
                    if 'edge_ts' in layer:
                        e_max = layer['edge_ts'].max().item()
                        if e_max > current_max:
                            leakages += 1
                            
            except Exception as e:
                print(f"  [Error reading {f.name}]: {e}")
                continue

        if violations == 0 and leakages == 0:
            print("  ✅ [Pass] No temporal violations or leakages detected.")
        else:
            print(f"  ❌ [Fail] Found {violations} order violations and {leakages} information leakages.")

    def eval_route_validity(self):
        """
        4. 路由覆盖率检查
        - 统计通信量，确保不是 0 (死路由)
        """
        print("\n🛣️ [Metric 4] Route Validity Check")
        
        chunk_dir = self.processed_dir_base / "part_0"
        files = list(chunk_dir.glob("slot_*.pt"))[:5] # 只检查前5个
        
        total_send = 0
        has_route = False
        
        for f in files:
            raw = torch.load(f, map_location='cpu')
            
            # 提取 comm_plans
            plans = []
            if hasattr(raw, 'comm_plans'): plans = raw.comm_plans
            elif isinstance(raw, list):
                # 尝试从 list[0] (task_data) 中找
                task = raw[0]
                plans = task.get('comm_plans', task.get('comm_plan', []))
            
            if not isinstance(plans, list): plans = [plans]
            
            for p in plans:
                if p is not None:
                    has_route = True
                    # send_sizes: [world_size]
                    if hasattr(p, 'send_sizes'):
                        total_send += p.send_sizes.sum().item()
        
        print(f"  - Route Object Found: {has_route}")
        print(f"  - Sampled Communication Volume: {int(total_send)} items")
        
        if total_send == 0 and has_route:
            print("  ⚠️ [Warning] Route exists but communication volume is ZERO. (Is this a single-partition run?)")
        elif total_send > 0:
            print("  ✅ [Pass] Valid communication traffic detected.")

    def run(self):
        print(f"🚀 Starting Evaluation for {self.dataset} ({self.num_parts} Partitions)")
        
        # 1. Load Balance
        total_stored = self.eval_load_balance()
        
        # 2. Communication
        self.eval_communication(total_stored)
        
        # 3. Temporal
        self.eval_temporal_integrity()
        
        # 4. Route
        self.eval_route_validity()
        
        print("\n✨ Evaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/mnt/data/zlj/starrygl-data/")
    parser.add_argument("--dataset", type=str, default="WIKI")
    parser.add_argument("--parts", type=int, default=4)
    args = parser.parse_args()
    
    evaluator = PreprocessingEvaluator(args.data_root, args.dataset, args.parts)
    evaluator.run()