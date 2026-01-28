import os
import sys
import argparse
import time
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset
import math
from functools import partial

# === Path Injection ===
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(str(parent_path))

from starrygl.cache.NodeState import DistNodeState, HistoryLayerUpdater
from starrygl.data.graph_context import StarryglGraphContext
from starrygl.utils.partition_book import PartitionState
from starrygl.data.batches import AtomicDataset, SlotAwareSampler, collate_and_merge
from starrygl.utils import DistributedContext, time_counter
from starrygl.cache.cache_route import CacheRouteManager
from starrygl.data.prefetcher import HostToDevicePrefetcher
from starrygl.nn.model.tgn import TGN 
from starrygl.nn.model.EdgePredictor import EdgePredictor
from starrygl.utils.params import get_config

# ==============================================================================
# Training Engine
# ==============================================================================

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
        self.full_dataset = AtomicDataset(self.processed_dir, neg_set=self.args.neg_set)
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
        subset = self.full_dataset.subset(indices)
        sampler = SlotAwareSampler(subset) # Compatible with Subset? Usually Subset changes indexing.
        # Check: SlotAwareSampler expects dataset to have __len__. Subset has it.
        # But indices inside Subset are remapped. 
        # SlotAwareSampler implementation usually iterates range(len(dataset)). 
        # So passing Subset is fine, it will yield 0..len(subset)-1, which Subset maps to real indices.
        
        ctx = DistributedContext.get_default_context()
        my_collate = partial(collate_and_merge, world_size=ctx.world_size)
        
        raw_loader = DataLoader(
            subset, batch_size=self.args.batch_slots, sampler=sampler,
            collate_fn=my_collate, num_workers=4, prefetch_factor=2, persistent_workers=True
        )
        return HostToDevicePrefetcher(
            loader=raw_loader, device=self.device, partition_state=self.partition_state,
            context=self.graph_context, hist_cache=self.history_states_updater
        )

    def prepare_model(self):
        mail_input = math.prod(self.graph_context.mailbox.shape[1:]) + self.cfg["train"]["dim_time"]
        self.router = CacheRouteManager.get_instance(self.ctx.group)
        hidden_dim = self.cfg["train"]["hidden_dim"]
        self.model = TGN(
            input_size=self.node_feat_cpu.shape[1], hidden_size=hidden_dim, output_size=hidden_dim,
            time_feats=self.cfg["train"]["dim_time"], edge_feats=self.dim_edge_feat,
            memory_params={'type': 'GRU', 'input_dim': mail_input},
            cache_manager=self.router, num_layers=2
        ).to(self.device)
        
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.ctx.rank], find_unused_parameters=True)
        self.predictor = EdgePredictor(hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.predictor.parameters()), lr=self.args.lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def reset_states(self):
        """Reset Memory and Mailbox for a new epoch."""
        pass
        # if hasattr(self.model.module, 'memory') and self.model.module.memory is not None:
        #     # Manually reset memory tensors if method doesn't exist
        #     if hasattr(self.model.module.memory, 'reset_memory'):
        #         self.model.module.memory.reset_memory()
        #     else:
        #         self.model.module.memory.memory.zero_()
        #         self.model.module.memory.last_update.zero_()
        
        # if hasattr(self.graph_context, 'mailbox'):
        #     self.graph_context.mailbox.reset()

    def run_epoch_step(self, loader, mode='train'):
        if mode == 'train': self.model.train()
        else: self.model.eval()
        
        total_loss = 0
        aps, aucs = [], []
        for step, batch in enumerate(loader):
            if batch is None: continue
            
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

                loss = self.criterion(
                    torch.cat([pos_out, neg_out]),
                    torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
                )

            if mode == 'train':
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
            if step % 20 == 0 and self.ctx.rank == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")
        return total_loss/(step+1), sum(aps)/len(aps) if aps else 0, sum(aucs)/len(aucs) if aucs else 0

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