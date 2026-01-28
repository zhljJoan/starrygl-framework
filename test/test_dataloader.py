import os
import sys
import argparse
import time
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import math
# === Path Injection ===
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(str(parent_path))

from starrygl.cache.NodeState import DistNodeState, HistoryLayerUpdater
from starrygl.data.graph_context import StarryglGraphContext
from starrygl.data.mailbox import DistMailbox
from starrygl.utils.partition_book import PartitionState
# === StarryGL Imports ===
from starrygl.data.batches import AtomicDataset, SlotAwareSampler, collate_and_merge
from starrygl.utils import DistributedContext, time_counter
from starrygl.cache.cache_route import CacheRouteManager

from starrygl.data.prefetcher import HostToDevicePrefetcher
from starrygl.nn.model.tgn import TGN 
from starrygl.nn.model.EdgePredictor import EdgePredictor

# [New] 引入现有的 Wrapper
from starrygl.nn.memory.mailbox import mailbox as Mailbox
from starrygl.utils.params import get_config
from functools import partial

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
        
        self.prepare_data()
        self.prepare_model()
        
    def prepare_data(self):
        root = Path(self.args.data_root)
        suffix = f"{self.args.dataset}_{self.ctx.size:03d}"
        meta_data = torch.load(root / "ctdg" / f"{self.args.dataset}.pth")
        num_nodes = meta_data['num_nodes']
        num_edges = meta_data['num_edges']
        # 1. 路径查找
        self.processed_dir = root / "processed_atomic" / suffix / f"part_{self.ctx.rank}"
        cand = list((root / "nparts").glob(f"{suffix}*"))
        if not cand: raise FileNotFoundError("Metadata not found")
        self.meta_dir = cand[0]
        
        # 2. 加载 Partition Metadata
        # partition_book: List[Tensor], index=rank -> [Owned, Halo]
        pb_data = torch.load(self.meta_dir / "partition_book.pt", map_location='cpu')
        rep_table = torch.load(self.meta_dir / "replica_table.pt", map_location='cpu')
        p_book = pb_data[0] if isinstance(pb_data, tuple) else pb_data
        
        # 本地维护的 ID 集合 (用于 PartitionState 判断 remote)
        self.local_ids = (pb_data[1] == torch.distributed.get_rank()).nonzero().squeeze()
        self.local_eids = (pb_data[2] == torch.distributed.get_rank()).nonzero().squeeze() if len(pb_data) > 2 else None

        self.partition_state = PartitionState(
            loc_ids = p_book[torch.distributed.get_rank()],
            num_master_nums = self.local_ids.shape[0] ,
            node_mode = 'map',
            node_replica_table= rep_table,
            loc_eids=self.local_eids,
            num_master_edges = self.local_eids.shape[0] if self.local_eids is not None else 0,
            edge_mode = 'map',
            num_nodes = num_nodes,
            num_edges = num_edges
            
        )

        
        self.ctx.sync_print("Initializing CPU States (Mailbox & History)...")
        feat_context = torch.load(self.meta_dir /f"part_{self.ctx.rank}"/"distributed_context.pt", map_location='cpu')
        self.edge_feat = feat_context.get('edge_feat', None)
        self.dim_edge_feat = self.edge_feat.shape[1] if self.edge_feat is not None else 0
        
        self.cfg = get_config(dataset = self.args.dataset, model = self.args.model)
        self.history_states = []
        for i in range(self.cfg["train"]["num_layers"]):
            memory_state = DistNodeState(
                    state = self.partition_state,
                    dim=self.cfg["train"]["hidden_dim"],
                )
            self.history_states.append(memory_state)
        if self.cfg['train']['memory_type'] is not None:
            memory_state = DistNodeState(
                state = self.partition_state,
                dim=self.cfg["train"]["hidden_dim"],
            )
            self.history_states.append(memory_state)
        else:
            self.history_states.append(None)
        self.history_states_updater = HistoryLayerUpdater(self.history_states)
        
        self.node_feat_cpu = feat_context.get('node_feat', None)
        if isinstance(self.node_feat_cpu, list):
            self.node_feat_cpu = torch.stack(self.node_feat_cpu, dim=0)
            self.node_feat_cpu = self.node_feat_cpu.permute(1,0,2).contiguous()
        self.graph_context = StarryglGraphContext(
            state = self.partition_state,
            node_feats = self.node_feat_cpu,
            edge_feats = self.edge_feat,
            mailbox_size=1, # TGN 通常只需要最新的，或者设大一点做序列
            dim_out=self.cfg["train"]["hidden_dim"], # Message 维度通常与 Memory 一致
            dim_edge_feat=self.dim_edge_feat
        )
        # 4. Loader
        self.dataset = AtomicDataset(self.processed_dir,neg_set = self.args.neg_set)
        self.sampler = SlotAwareSampler(self.dataset)
        ctx = DistributedContext.get_default_context()
        my_collate = partial(collate_and_merge, world_size=ctx.world_size)
        self.raw_loader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_slots,
            sampler=self.sampler,
            collate_fn=my_collate,
            # === 关键性能参数 ===
            num_workers=4,           # 建议设置为 CPU 物理核心数的一半
            prefetch_factor=2,       # 每个 worker 提前加载 2 个 batch
            persistent_workers=True, # [重要] 一个 Epoch 结束后不关闭进程，避免重建开销
            pin_memory=False         # 我们在 Prefetcher 里手动处理 pin，这里不需要
        )
        self.loader = HostToDevicePrefetcher(
            loader=self.raw_loader,
            device=self.device,
            partition_state=self.partition_state,
            context=self.graph_context,
            hist_cache=self.history_states_updater
        )
    def prepare_model(self):
        mail_input = math.prod(self.graph_context.mailbox.shape[1:]) + self.cfg["train"]["dim_time"]
        self.router = CacheRouteManager.get_instance(self.ctx.group)
        hidden_dim = self.cfg["train"]["hidden_dim"]
        self.model = TGN(
            input_size=self.node_feat_cpu.shape[1],
            hidden_size=hidden_dim,
            output_size=hidden_dim,
            time_feats=self.cfg["train"]["dim_time"],
            edge_feats=self.dim_edge_feat,
            memory_params={'type': 'GRU', 'input_dim': mail_input},
            cache_manager=self.router,
            num_layers=2
        ).to(self.device)
        
        self.model = nn.parallel.DistributedDataParallel(
            self.model, device_ids=[self.ctx.rank], find_unused_parameters=True
        )
        self.predictor = EdgePredictor(hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.predictor.parameters()), lr=self.args.lr
        )
        self.criterion = nn.BCEWithLogitsLoss()


    def train_epoch(self):
        self.model.train()
        total_loss = 0
        loader = self.loader
        aps = []
        start = time_counter.time_count.start()
        for step, batch in enumerate(loader):
            if batch is None: continue
            
            # TGN 内部做 Layer-wise Sync
            h = self.model(
                blocks=batch.mfgs,
                routes=batch.routes,
                mailbox_data=batch.mailbox,
                upd_hook = self.history_states_updater.update_embedding_and_broadcast
            )

            # Loss
            task = batch.roots
            num_src = task['task_src'].shape[0]
            inv_map = task.get('inv_map') if 'inv_map' in task else None
            src_emb = h[inv_map[:num_src]] if inv_map is not None else h[:num_src]
            dst_emb = h[inv_map[num_src:2*num_src]] if inv_map is not None else h[num_src:2*num_src]
            neg_emb = h[inv_map[2*num_src:]] if (inv_map is not None and 'task_neg_dst' in task) else None

            if neg_emb is not None:
                pos_out, neg_out = self.predictor(src_emb, dst_emb, h_neg_dst=neg_emb)
            else:
                pos_out = self.predictor(src_emb, dst_emb)
                neg_out = torch.zeros_like(pos_out)
                
            loss = self.criterion(
                torch.cat([pos_out, neg_out]),
                torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.cfg["train"]["memory_type"] is not None:
                with torch.no_grad():
                    new_mem = self.model.module.last_memory
                    new_mem_ts = self.model.module.last_ts
                    self.graph_context.mailbox.generate_message_and_update(
                        task,
                        new_mem,
                        new_mem_ts,
                        self.graph_context
                    ) 
            total_loss += loss.item()
            y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0)
            aps.append(average_precision_score(y_true, y_pred.detach().numpy()))
            if step % 20 == 0 and self.ctx.rank == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")
                
        end = time_counter.time_count.elapsed_event(start)
        if self.ctx.rank == 0:
            print(f"Avg Loss: {total_loss / (step + 1):.4f} | Avg AP: {sum(aps) / len(aps):.4f} | Time: {end:.4f}s")
        return total_loss

    def run(self):
        for ep in range(self.args.epochs):
            self.train_epoch()
            # 可以在这里做 Checkpoint

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError: pass
    
    TrainingEngine().run()