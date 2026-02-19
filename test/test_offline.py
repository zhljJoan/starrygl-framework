import os
import sys
import argparse
import time
import psutil
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset
import math
from functools import partial
from torch.profiler import profile, record_function, ProfilerActivity, schedule
# === Path Injection ===
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(str(parent_path))
import torch.cuda.nvtx as nvtx
from starrygl.cache.NodeState import DistNodeState, HistoryLayerUpdater
from starrygl.data.graph_context import StarryglGraphContext
from starrygl.utils.partition_book import PartitionState
from starrygl.data.batches import CTDGBatchSampler,  StarryGLDataset, collate_and_merge
from starrygl.utils import DistributedContext, time_counter
from starrygl.cache.cache_route import CacheRouteManager
from starrygl.data.prefetcher import HostToDevicePrefetcher, OptimizedPrefetcher, ThreadedPrefetcher
from starrygl.nn.model.tgn import TGN 
from starrygl.nn.model.EdgePredictor import EdgePredictor
from starrygl.utils.params import get_config
import torchvision.models as models
# ==============================================================================
# Training Engine
# ==============================================================================
def setup_numa_affinity(gpu_id=0, num_main_cores=4, num_prefetch_cores=4):
    """
    根据 GPU ID 自动设置 NUMA 亲和性，隔离主线程和预取线程
    """
    total_logical_cores = psutil.cpu_count()
    if gpu_id in [0, 1]:
        candidate_cores = list(range(0, total_logical_cores, 2))
        node_name = "NUMA Node 0 (Even)"
    elif gpu_id in [2, 3]:
        candidate_cores = list(range(1, total_logical_cores, 2))
        node_name = "NUMA Node 1 (Odd)"
    else:
        print(f"[Warn] Unknown GPU ID {gpu_id}, no affinity set.")
        return [], []

    print(f"=== Affinity Setup for GPU {gpu_id} ({node_name}) ===")
    offset = 4 
    available = candidate_cores[offset:] 
    
    if len(available) < (num_main_cores + num_prefetch_cores):
        print("[Warn] Not enough cores for strict isolation!")
        available = candidate_cores
    main_cores = available[:num_main_cores]
    bg_start_idx = num_main_cores + 2
    prefetch_cores = available[bg_start_idx : bg_start_idx + num_prefetch_cores]
    print(f" -> Main Thread Cores : {main_cores}")
    print(f" -> Prefetcher Cores  : {prefetch_cores}")
    p = psutil.Process()
    try:
        p.cpu_affinity(main_cores)
    except:
        os.sched_setaffinity(0, main_cores)
    torch.set_num_threads(len(main_cores))    
    return main_cores, prefetch_cores

class TrainingEngine:
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default="WIKI")
        parser.add_argument("--data-root", type=str, default="/mnt/data/zlj/starrygl-data/")
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--batch-slots", type=int, default=4)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--model", type=str, default="tgn", help="model")
        parser.add_argument("--neg_set", type=int, default=8, help="number of negative sets")
        return parser.parse_args()
    
    def __init__(self):
        self.args = self.parse_args()
        self.ctx = DistributedContext.init("nccl")
        self.device = self.ctx.device
        self.cfg = get_config(dataset=self.args.dataset, model=self.args.model) # Early config load
        
        self.prepare_data()
        self.prepare_model()
        
    def prepare_data(self):
        root = Path(self.args.data_root)
        suffix = f"{self.args.dataset}_{self.ctx.size:03d}"
        
        # Metadata
        self.processed_dir = root / "processed_atomic" / suffix / f"part_{self.ctx.rank}"
        cand = list((root / "nparts").glob(f"{suffix}*"))
        if not cand: raise FileNotFoundError("Metadata not found")
        self.meta_dir = cand[0]
        
        pb_data = torch.load(self.meta_dir / "partition_book.pt", map_location='cpu')
        rep_table = torch.load(self.meta_dir / "replica_table.pt", map_location='cpu')
        p_book = pb_data[0] if isinstance(pb_data, tuple) else pb_data
        
        self.local_ids = (pb_data[1] == torch.distributed.get_rank()).nonzero().squeeze()
        self.local_eids = (pb_data[2] == torch.distributed.get_rank()).nonzero().squeeze() if len(pb_data) > 2 else None
        
        num_nodes_total = torch.load(root / "ctdg" / f"{self.args.dataset}.pth")['num_nodes']
        num_edges_total = torch.load(root / "ctdg" / f"{self.args.dataset}.pth")['num_edges']

        self.partition_state = PartitionState(
            loc_ids = p_book[torch.distributed.get_rank()],
            num_master_nums = self.local_ids.shape[0] ,
            node_mode = 'map', node_replica_table= rep_table,
            loc_eids=self.local_eids,
            num_master_edges = self.local_eids.shape[0] if self.local_eids is not None else 0,
            edge_mode = 'map', num_nodes = num_nodes_total, num_edges = num_edges_total
        )
        
        # History
        self.history_states = []
        for i in range(self.cfg["train"]["num_layers"]):
            self.history_states.append(DistNodeState(state=self.partition_state, dim=self.cfg["train"]["hidden_dim"]))
        
        if self.cfg['train']['memory_type'] is not None:
            self.history_states.append(DistNodeState(state=self.partition_state, dim=self.cfg["train"]["hidden_dim"]))
        else:
            self.history_states.append(None)
        self.history_states_updater = HistoryLayerUpdater(self.history_states)
        
        # Features
        feat_context = torch.load(self.meta_dir /f"part_{self.ctx.rank}"/"distributed_context.pt", map_location='cpu')
        self.edge_feat = feat_context.get('edge_feat', None)
        self.dim_edge_feat = self.edge_feat.shape[1] if self.edge_feat is not None else 0
        self.node_feat_cpu = feat_context.get('node_feat', None)
        if isinstance(self.node_feat_cpu, list):
            self.node_feat_cpu = torch.stack(self.node_feat_cpu, dim=0).permute(1,0,2).contiguous()

        self.graph_context = StarryglGraphContext(
            state=self.partition_state, node_feats=self.node_feat_cpu, edge_feats=self.edge_feat,
            mailbox_size=1, dim_out=self.cfg["train"]["hidden_dim"], dim_edge_feat=self.dim_edge_feat
        )

        # === [核心逻辑] Dataset Split (70/15/15) ===
        self.full_dataset = StarryGLDataset(self.processed_dir)
        #AtomicDataset(self.processed_dir, neg_set=self.args.neg_set)
        total_len = len(self.full_dataset)
        train_len = int(total_len * 0.70)
        val_len = int(total_len * 0.15)
        test_len = total_len - train_len - val_len
        
        all_indices = list(range(total_len))
        self.train_idx = slice(0, train_len)
        self.val_idx = slice(train_len, train_len + val_len)
        self.test_idx = slice(train_len + val_len, total_len)
        
        #self.ctx.sync_print(f"Split: Train={len(self.train_idx)}, Val={len(self.val_idx)}, Test={len(self.test_idx)}")

    def _create_loader(self, indices:slice):
        #if len(indices) == 0: return None
        subset = self.full_dataset
        sampler = CTDGBatchSampler(subset, batch_size = self.args.batch_slots, subset_range=indices) # Compatible with Subset? Usually Subset changes indexing.
        ctx = DistributedContext.get_default_context()
        my_collate = partial(collate_and_merge, world_size=ctx.world_size)
        
        raw_loader = DataLoader(
            subset, batch_sampler=sampler, collate_fn=my_collate, num_workers=1, prefetch_factor=2, 
            persistent_workers=True, pin_memory=True
        )
        original_prefetcher = OptimizedPrefetcher(#HostToDevicePrefetcher(
            loader=raw_loader, device=self.device, partition_state=self.partition_state,
            context=self.graph_context, hist_cache=self.history_states_updater
        )
        #return original_prefetcher
        return original_prefetcher#ThreadedPrefetcher(original_prefetcher)

    def prepare_model(self):
        mail_input = math.prod(self.graph_context.mailbox.shape[1:]) + self.cfg["train"]["dim_time"]
        self.router = CacheRouteManager.get_instance(self.ctx.group)
        hidden_dim = self.cfg["train"]["hidden_dim"]
        self.model = TGN(
            input_size=self.node_feat_cpu.shape[1], hidden_size=hidden_dim, output_size=hidden_dim,
            time_feats=self.cfg["train"]["dim_time"], edge_feats=self.dim_edge_feat,
            memory_params={'type': 'GRU', 'input_dim': mail_input},
            cache_manager=self.router, num_layers=self.cfg["train"]["num_layers"],
            num_heads=self.cfg["train"]["num_heads"],
            dropout=self.cfg["train"]["dropout"], att_dropout=self.cfg["train"]["att_dropout"]
        ).to(self.device)
        
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.ctx.rank], find_unused_parameters=True)
        #self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=True)
        self.predictor = EdgePredictor(hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.predictor.parameters()), lr=self.args.lr, fused=True)
        self.criterion = nn.BCEWithLogitsLoss()

    def reset_states(self):
        """Reset Memory and Mailbox for a new epoch."""
        #pass
        self.history_states_updater.reset()
        if self.cfg["train"]["memory_type"] is not None:
            self.graph_context.mailbox.reset()
        # if hasattr(self.model.module, 'memory') and self.model.module.memory is not None:
        #     # Manually reset memory tensors if method doesn't exist
        #     if hasattr(self.model.module.memory, 'reset_memory'):
        #         self.model.module.memory.reset_memory()
        #     else:
        #         self.model.module.memory.memory.zero_()
        #         self.model.module.memory.last_update.zero_()
        
        # if hasattr(self.graph_context, 'mailbox'):
        #        self.graph_context.mailbox.reset()

    def run_epoch_step(self, loader, mode='train'):
        if mode == 'train': self.model.train()
        else: self.model.eval()
        
        total_loss = 0
        aps, aucs = [], []
        print(f"Running epoch step in {mode} mode...")
        for step, batch in enumerate(loader):
            if batch is None: continue
            nvtx.range_push(f"Step Forward")
            with torch.set_grad_enabled(mode == 'train'):
                h = self.model(
                    blocks=batch.mfgs, routes=batch.routes, mailbox_data=batch.mailbox,
                    upd_hook=self.history_states_updater.update_embedding_and_broadcast
                )
                task = batch.roots
                num_src = task['task_src'].shape[0]
                inv_map = task.get('inv_map')
                
                src_emb = h[inv_map[:num_src]]
                dst_emb = h[inv_map[num_src:2*num_src]]
                neg_emb = h[inv_map[2*num_src:]] if 'task_neg_dst' in task else None
                
                if neg_emb is not None:
                    pos_out, neg_out = self.predictor(src_emb, dst_emb, h_neg_dst=neg_emb)
                else:
                    pos_out = self.predictor(src_emb, dst_emb)
                    neg_out = torch.zeros_like(pos_out)

                #loss = self.criterion(
                #    torch.cat([pos_out, neg_out]),
                #    torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
                #)
                loss = torch.nn.functional.softplus(-pos_out).mean() + torch.nn.functional.softplus(neg_out).mean()
            nvtx.range_pop()
            nvtx.range_push(f"Step Backward")
            if mode == 'train':
                #pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # [Important] Update memory in ALL modes (Train/Val/Test) to maintain state
            if self.cfg["train"]["memory_type"] is not None:
                with torch.no_grad():
                    new_mem = self.model.module.last_memory
                    new_mem_ts = self.model.module.last_ts
                    self.graph_context.mailbox.generate_message_and_update(
                        task, new_mem, new_mem_ts, self.graph_context
                    )

            total_loss += loss.item()
            with torch.no_grad():
                y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
                y_true = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0).cpu()
                aps.append(average_precision_score(y_true, y_pred))
                try: aucs.append(roc_auc_score(y_true, y_pred))
                except: pass
            
            nvtx.range_pop()
            if step % 20 == 0 and self.ctx.rank == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")
        return total_loss/(step+1), sum(aps)/len(aps) if aps else 0, sum(aucs)/len(aucs) if aucs else 0

    
    # def run_epoch_step(self, loader, mode='train'):
    #     if mode == 'train': self.model.train()
    #     else: self.model.eval()
        
    #     total_loss = 0
    #     aps, aucs = [], []
    #     print(f"Running epoch step in {mode} mode...")

    #     # === [新增] 定义 Profiler Schedule ===
    #     # wait=1: 跳过第1个step (通常主要是数据加载初始化)
    #     # warmup=1: 第2个step预热 (不记录)
    #     # active=3: 记录第3,4,5个step (这就是你会看到的分析数据)
    #     # repeat=1: 只执行一轮这样的循环
    #     my_schedule = schedule(wait=1, warmup=1, active=3, repeat=1)

    #     # 定义 trace 处理器，保存 JSON
    #     def trace_handler(p):
    #         output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    #         print(output)
    #         # 文件名包含 rank 以区分多卡
    #         p.export_chrome_trace(f"trace_rank{self.ctx.rank}.json")
    #         print(f"Profiler trace saved to trace_rank{self.ctx.rank}_1.json")
    #     @torch.jit.script
    #     def fused_loss_calc(pos_out: torch.Tensor, neg_out: torch.Tensor) -> torch.Tensor:
    #         # JIT 会把这些操作融合在一两个 Kernel 里，并省去 Python 调度
    #         return torch.nn.functional.softplus(-pos_out).mean() + torch.nn.functional.softplus(neg_out).mean()

    #     # === [新增] 开启 Profiler ===
    #     # 仅在 train 模式且 rank 0 时开启 (避免生成太多文件，除非你想看所有卡的表现)
    #     # 如果你想看所有卡，去掉 "if self.ctx.rank == 0 else None" 的判断
    #     # tb_handler = torch.profiler.tensorboard_trace_handler(
    #     #     dir_name=f"./log/profiler_rank{self.ctx.rank}",
    #     #     use_gzip=True # 压缩日志，节省空间
    #     # )

    #     # 2. 开启 Profiler
    #     use_profiler = False#(mode == 'train' and self.ctx.rank == 0)
        
    #     # 创建一个上下文管理器（如果是 eval 模式或者是其他 rank，就是一个空的上下文）
    #     prof_ctx = profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         schedule=my_schedule,
    #         on_trace_ready=trace_handler,
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True # 关键：显示代码行号
    #     ) if use_profiler else torch.autograd.detect_anomaly(False) # 这里的 else 只是为了语法占位，实际可以用 nullcontext

    #     with prof_ctx as prof:
    #         for step, batch in enumerate(loader):
    #             # 为了快速获得 Profile，我们不需要跑完 400 步
    #             # 只要跑完 active 的那几步 (wait+warmup+active = 1+1+3 = 5步) 就可以拿到数据了
    #             if step >= 6 and use_profiler: 
    #                 print("Profiling finished, stopping early for analysis.")
    #                 break 
                
    #             #if(step>400): break
    #             if batch is None: continue

    #             # NVTX 标记会被 Profiler 自动捕获，保留即可
    #             nvtx.range_push(f"Step Forward")
                
    #             with torch.set_grad_enabled(mode == 'train'):
    #                 # --- 模型前向 ---
    #                 h = self.model(
    #                     blocks=batch.mfgs, routes=batch.routes, mailbox_data=batch.mailbox,
    #                     upd_hook=self.history_states_updater.update_embedding_and_broadcast
    #                 )
                    
    #                 task = batch.roots
    #                 num_src = task['task_src'].shape[0]
    #                 inv_map = task.get('inv_map')
    #                 src_emb = h[inv_map[:num_src]]
    #                 dst_emb = h[inv_map[num_src:2*num_src]]
    #                 neg_emb = h[inv_map[2*num_src:]] if 'task_neg_dst' in task else None
    #                 pos_out, neg_out = self.predictor(src_emb, dst_emb, h_neg_dst=neg_emb)
    #                 loss = torch.nn.functional.softplus(-pos_out).mean() + torch.nn.functional.softplus(neg_out).mean()
    #                 #fused_loss_calc(pos_out, neg_out)
    #                 #loss = self.criterion(
    #                 #    torch.cat([pos_out, neg_out]),
    #                 #    torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
    #                 #)
    #             nvtx.range_pop() # End Forward

    #             nvtx.range_push(f"Step Backward")
    #             if mode == 'train':
    #                 self.optimizer.zero_grad()
    #                 loss.backward()
    #                 self.optimizer.step()
                
    #             # Update memory
    #             if self.cfg["train"]["memory_type"] is not None:
    #                 with torch.no_grad():
    #                     new_mem = self.model.last_memory
    #                     new_mem_ts = self.model.last_ts
    #                     self.graph_context.mailbox.generate_message_and_update(
    #                         task, new_mem, new_mem_ts, self.graph_context
    #                     )

    #             total_loss += loss.item()
    #             # with torch.no_grad():
    #             #     # 暂时注释掉 metrics 计算以减少 Profile 噪音，除非你特别想看 metrics 的耗时
    #             #     y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
    #             #     y_true = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0).cpu()
    #             #     aps.append(average_precision_score(y_true, y_pred))
                
    #             nvtx.range_pop() # End Backward

    #             if step % 100 == 0 and self.ctx.rank == 0:
    #                 print(f"Step {step} | Loss: {loss.item():.4f}")
                
    #             # === [关键] 通知 Profiler 进入下一步 ===
    #             if use_profiler:
    #                 prof.step()

    #     return total_loss/(step+1), 0, 0 # 临时返回 0,0 避免计算错误
    def run(self):
        # Create loaders lazily or upfront
        print('rank{} train_idx {} val_idx {} test_idx {}'.format(self.ctx.rank, self.train_idx, self.val_idx, self.test_idx))
        train_loader = self._create_loader(self.train_idx)
        val_loader = self._create_loader(self.val_idx)
        test_loader = self._create_loader(self.test_idx)
        
        for ep in range(self.args.epochs):
            start = time_counter.time_count.start()
            
            # 1. Reset State at start of Epoch
            self.reset_states()
            if hasattr(self.full_dataset, 'set_epoch'):
                self.full_dataset.set_epoch(ep)
            
            # 2. Train
            t_loss, t_ap, t_auc = self.run_epoch_step(train_loader, mode='train')
            end = time_counter.time_count.elapsed_event(start)
            # 3. Val (Continue state from Train)
            v_loss, v_ap, v_auc = self.run_epoch_step(val_loader, mode='eval')
            
            if self.ctx.rank == 0:
                print(f"Ep {ep} | T: {end:.2f}s | Train L:{t_loss:.4f} AP:{t_ap:.4f} | Val L:{v_loss:.4f} AP:{v_ap:.4f}")

        # 4. Final Test
            if self.ctx.rank == 0: print("\n=== Final Testing ===")
        # Note: Ideally you should reload best model and re-run [Train -> Val -> Test] to establish memory
        # Here we just run Test after the last Epoch's Val, which is acceptable for inductive eval
            test_loss, test_ap, test_auc = self.run_epoch_step(test_loader, mode='eval')
            if self.ctx.rank == 0:
                print(f"Test Result | Loss: {test_loss:.4f} | AP: {test_ap:.4f} | AUC: {test_auc:.4f}")

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError: pass
    TrainingEngine().run()