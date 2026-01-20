import os
import sys
from os.path import abspath, join, dirname
from typing import List, Optional, Tuple

from sklearn.metrics import average_precision_score
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from pathlib import Path
import math

# StarryGL 模块
from starrygl.cache.replica_table import CSRReplicaTable, UVACSRReplicaTable
from starrygl.graph.graph import pyGraph, AsyncGraphBlob, AsyncBlock
from starrygl.data.chunk_dataloader import AsyncPipelineLoader, NegativeSetSampler, StarryGLDataset, StarryBatchData
# 假设 GraphCollator 已经在 starrygl.data.chunk_dataloader 中更新为支持 mailbox/history 的版本
# 如果没有，请使用之前对话中提供的 updated GraphCollator 代码替换
from starrygl.data.chunk_dataloader import GraphCollator 

from starrygl.route.route import DistRouteIndex, PartitionState, Route
from starrygl.utils import DistributedContext
from starrygl.utils.params import *
from starrygl.utils.parser import parse_chunk_decay
from starrygl.data import PartitionData, STGraphLoader, STGraphBlob
from starrygl.cache.NodeState import HistoryLayerUpdater, NodeState, CachedNodeState

# 引入 TGN 相关模块 (假设文件已上传或在路径中)
from starrygl.nn.memory.mailbox import mailbox as Mailbox
from starrygl.nn.model.tgn import TGN
from starrygl.nn.model.EdgePredictor import EdgePredictor

# 简单的负采样器实现
class NegativeSampler:
    def __init__(self, num_nodes, device):
        self.num_nodes = num_nodes
        self.device = device
    
    def sample(self, num_samples):
        return torch.randint(0, self.num_nodes, (num_samples,), device=self.device, dtype=torch.long)

class TrainingEngine:
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)
        parser.add_argument("--dataset", type=str, required=True)#, choices=DATASET_CHOICES)
        parser.add_argument("--data-root", type=str, default="/mnt/data/zlj/starrygl-data/")
        parser.add_argument("--epochs", type=int, default=200)
        parser.add_argument("--learning-rate", "--lr", dest="lr", type=float, default=1e-3)
        parser.add_argument("--fulls-count", type=int, default=2)
        parser.add_argument("--load_full_sample_graph", action = "store_true", default = False)
        parser.add_argument("--chunk-size", type=int, default=16)
        parser.add_argument("--hot_ratio", type=float, default=0.1)
        parser.add_argument("--add_inverse_edge", action="store_true", default=False)
        
        return parser.parse_args()
    
    def __init__(self, args=None):
        if args is None:
            args = self.parse_args()
        self.args = args

        self.ctx = DistributedContext.init("nccl")
        
        # 1. 加载数据与状态
        self.load_dataset()
        
        # 2. 加载模型
        self.load_model()
        
    def load_dataset(self):
        self.data_root: Path = Path(self.args.data_root).expanduser().resolve()
        self.data_name: str = self.args.dataset
        
        # --- 加载 PartitionData ---
        path = self.data_root / "processed" / f"{self.data_name}" / f"{self.ctx.size:03d}_{self.args.chunk_size:03d}_{self.args.hot_ratio:0.1f}_{self.ctx.rank+1:03d}.pth"
        lookup_table = self.data_root / "processed" / f"{self.data_name}" / f"{self.ctx.size:03d}_{self.args.hot_ratio:0.1f}_lookup_table.pth"
        
        self.ctx.sync_print(f"loading lookup table from {lookup_table}...")
        self.lookup_table = torch.load(lookup_table, map_location="cpu")
        
        partition_book = CSRReplicaTable(
            indptr=self.lookup_table['indptr'],
            indices=self.lookup_table['indices'],
            locs=self.lookup_table['locs'],
            device=self.ctx.device
        )
        self.cfg = get_config(dataset = self.data_name, model = self.args.model)
        self.ctx.sync_print(f"loading data from {path}...")
        self.data = PartitionData.load(path) # Keep raw data accessible
        
        # 构建 PartitionState
        self.partition_state = PartitionState(
            loc_ids = self.data.node_data['nid'][0].data ,
            loc_eids = self.data.edge_data['eid'].data,
            is_shared = DistRouteIndex(self.data.node_data['NID'][0].data).is_shared,
            partition_book=partition_book,
            dist_nid_mapper= self.lookup_table['distNID'].data,
            dist_eid_mapper=self.lookup_table['distEID'].data
        )

        # 注册 Chunk 缓存
        self.chunk_index = self.data.pop_ndata('c')[0].item()
        self.data.register_to_chunk_cached_data(node_chunk_id = self.chunk_index)
        
        # 边数据的 chunk 归属 (由 src 决定)
        edge_src_chunk = self.chunk_index[self.data.edge_src.data]
        self.data.register_to_edge_chunk_cached_data(node_chunk_id = edge_src_chunk)

        # --- 初始化 NodeState (History & Memory) ---
        # 假设 hidden_dim 从 args 获取
        dim_hidden = self.cfg["train"]["hidden_dim"]
        num_local_nodes = self.chunk_index.numel()
        
        # History State: 存储 GNN 输出的历史嵌入 (用于 Stale 补偿)
        # 这里可以使用 CachedNodeState 以利用 GPU 缓存
        self.history_states = []
        for i in range(self.cfg["train"]["num_layers"]):
            memory_state = CachedNodeState(
                data=NodeState(
                    node_nums=num_local_nodes,
                    dim=self.cfg["train"]["hidden_dim"],
                    partition_state=self.partition_state,
                    device=self.ctx.device
                ),
                node_chunk_id=self.chunk_index,
                device=self.ctx.device
            ) 
            self.history_states.append(memory_state)
        self.history_states_updater = HistoryLayerUpdater(self.history_states)
        #self.history_state = HistoryLayerUpdater(history_states)
        # --- 初始化 Mailbox ---
        # Mailbox 存储最近的交互信息
        # 维度: 2 * dim_embed (msg) + dim_edge_feat
        dim_edge_feat = self.data.edge_data['f'].data.shape[1]
        self.mailbox = Mailbox(
            num_nodes=num_local_nodes,
            mailbox_size=1, # TGN 通常只需要最新的，或者设大一点做序列
            dim_out=self.cfg["train"]["hidden_dim"], # Message 维度通常与 Memory 一致
            dim_edge_feat=dim_edge_feat,
            device=self.ctx.device
        )

        self.chunk_count = self.chunk_index.max().item() + 1
        self.ctx.sync_print(f"chunk_count = {self.chunk_count}, local_nodes = {num_local_nodes}")

        # --- 初始化 Graph Engine (pyGraph) ---
        num_nodes = torch.tensor([self.chunk_index.numel()], dtype=torch.long, device="cpu").item()
        print('num_nodes:{} check: {} {}\n'.format(num_nodes, self.data.edge_src.data.max().item(), self.data.edge_dst.data.max().item()))
        
        ctx = DistributedContext.get_default_context()
        self.graph_stream = torch.cuda.Stream(device=ctx.device)
        if(self.args.add_inverse_edge):
            print('add inverse')
            eid = torch.arange(self.data.edge_src.data.size(0)*2, dtype=torch.long, device=ctx.device)
            src = torch.cat([self.data.edge_src.data.reshape(-1,1), self.data.edge_dst.data.reshape(-1,1)], dim=1).reshape(-1)
            dst = torch.cat([self.data.edge_dst.data.reshape(-1,1), self.data.edge_src.data.reshape(-1,1)], dim=1).reshape(-1)
            ts = torch.cat([self.data.edge_data['ts'].data.reshape(-1,1), self.data.edge_data['ts'].data.reshape(-1,1)], dim=1).reshape(-1)
            
        else:
            src = self.data.edge_src.data
            dst = self.data.edge_dst.data
            ts = self.data.edge_data['ts'].data
            eid = torch.arange(self.data.edge_src.data.size(0), dtype=torch.long, device=ctx.device)
        g = pyGraph(
            src = src.long(),
            dst = dst.long(),
            ts = ts.long(),
            eid = eid.long(),
            node_num = num_nodes,
            chunk_size = self.chunk_count,
            chunk_mapper = self.chunk_index.long(),
            stream = self.graph_stream.cuda_stream
        )
        event_chunk_id = self.chunk_index[src]
        
        self.dataset = StarryGLDataset(
            node_chunk_id=self.chunk_index, 
            ts=ts.long(), 
            event_chunk_id=event_chunk_id, # [New]
            rank=dist.get_rank(), 
            world_size=dist.get_world_size()
        )
        
        # 划分训练集
        train_ratio = 0.7
        self.train_end_ts = ts.max().item() * train_ratio
        
        # Negative Sampler
        self.neg_sampler = NegativeSetSampler(nodes=self.data.edge_dst.data[:dst.numel()//2])
        
        # Collator
        self.collator = GraphCollator(
            partition_data=self.data,
            history_state=self.history_states, # 这里作为 List 传入，对应每一层? 暂时只传一个对象
            mailbox=self.mailbox,
            partition_state=self.partition_state,
            device=self.ctx.device,
            self_loop=self.cfg["train"]["self_loop"],
            add_inverse_edge=self.args.add_inverse_edge, # [New] 添加逆边支持
        )
        
        
        self.train_loader = AsyncPipelineLoader(
            dataset=self.dataset, # 需要 dataset 支持切片或在 iter 中控制 range
            graph_engine= g,
            collator=self.collator,
            sampling_config=self.cfg["sample"],
            partition_data=self.data,
            mode='CTDG',
            prefetch_factor=2,
            negative_sampler=self.neg_sampler # 传给 loader 以便 task 生成时使用
        )
 
    def load_model(self):
        # 参数准备
        dim_node = self.data.node_data['f'].data.shape[1]
        dim_edge = self.data.edge_data['f'].data.shape[1]
        
        memory_param = {
            'type': self.cfg['train']['memory_type'],
            'input_dim': self.cfg['train']['hidden_dim'] * 2 + dim_edge, # Msg = Mem_Src + Mem_Dst + Edge
            'memory_dim': self.cfg['train']['hidden_dim'],
            'time_dim': self.cfg['train']['dim_time']
        }
        
        # TGN 模型
        self.model = TGN(
            input_size=dim_node,
            hidden_size=self.cfg['train']['hidden_dim'],
            output_size=self.cfg['train']['hidden_dim'],
            time_feats=self.cfg['train']['dim_time'],
            edge_feats=dim_edge,
            memory_params=memory_param,
            num_layers=self.cfg['train']['num_layers'], # 对应采样层数
            num_heads=self.cfg['train']['num_heads'],
            dropout=self.cfg['train']['dropout'],
            att_dropout=self.cfg['train']['att_dropout']
        ).to(self.ctx.device)
        
        # 边预测器
        self.predictor = EdgePredictor(self.cfg['train']['hidden_dim']).to(self.ctx.device)
        
        # 优化器 & Loss
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.predictor.parameters()), 
            lr=self.cfg['train']['lr']
        )
        self.criterion = nn.BCEWithLogitsLoss()

        # DDP (如果需要)
        self.model = nn.parallel.DistributedDataParallel(self.model)

    def train_epoch(self):
        self.model.train()
        self.predictor.train()
        
        loss_acc, loss_cnt = 0.0, 0
    
        def generate_original_events(embding, root, root_ptr, ts, ptrs, eids):
            repeat = ptrs[1:] - ptrs[:-1]
            roots = root.repeat_interleave(root_ptr[1:]-root_ptr[:-1], dim=0).repeat_interleave(repeat, dim=0)
            ts = ts.repeat_interleave(repeat, dim=0)
            emb = embding[ptrs[:-1]].repeat_interleave(repeat, dim=0)
            q_ids = eids//2
            src_mask = eids%2
            src_ids = q_ids[src_mask]
            dst_ids = q_ids[~src_mask]
            sort_eid, pos = torch.sort(src_ids)
            src_emb = emb[src_mask][pos]
            src = roots[src_mask][pos]
            _, pos = torch.sort(dst_ids)
            dst_emb = emb[~src_mask][pos]
            dst = roots[~src_mask][pos]
            ts = ts[src_mask][pos]
            return src_emb, dst_emb, sort_eid, src, dst, ts
            
            
        # 遍历 Loader
        ap = []
        for batch in self.train_loader:
            if batch is None: break
            h_all = self.model(
                g=batch.mfgs,
                dist_flag=batch.dist_flag,
                history=batch.history,
                memory=batch.memory,
                mailbox=batch.mailbox,
                nid_mapper=batch.nid_mapper,
                upd_hook=self.history_states_updater
            )
            root,neg_root = batch.roots
            
            num_events = batch.roots[0].size(0)
            pos_num = root.ts.numel()
            src_emb, dst_emb, sort_eid, src, dst, ts = generate_original_events(h_all[:pos_num], root, root.q_ptr, root.ts, root.q_eid)
            neg_dst_emb = h_all[pos_num:]
            pos_score, neg_score = self.predictor(src_emb, dst_emb, neg_dst_emb = neg_dst_emb, neg_samples=1, mode='triplet')
            y_true = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)
            y_pred = torch.cat([pos_score, neg_score], dim=0)
            loss = self.criterion(
                y_pred,
                y_true
            )
            # 4. Backward
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            with torch.no_grad():
                # Update Src's Mailbox with (Dst, Time, Feat)
                if self.model.last_memory is not None:  
                    edge_feat = self.data.edge_data['f'].select(
                        sort_eid, 0
                    ).to(self.ctx.device)
                    src_mem = self.model.last_memory[src]
                    dst_mem = self.model.last_memory[dst]
                    src = batch.nid_mapper[src]
                    dst = batch.nid_mapper[dst]
                    index, mail, mail_ts = self.mailbox.get_update_mail(src, dst, ts, edge_feat, src_mem, dst_mem)
                    self.mailbox.update_mailbox(index, mail, mail_ts)
            
            # 打印日志
                loss_acc += loss.item()
                ap.append(average_precision_score(y_true.cpu(), y_pred.cpu()))
                loss_cnt += 1
                if loss_cnt % 10 == 0:
                    print(f"Batch {loss_cnt}, Loss: {loss.item():.4f}, ap: {sum(ap)/len(ap):.4f}")

        return loss_acc / loss_cnt, sum(ap) / len(ap) if ap else 0.0
    
    def run(self):
        self.ctx.init_disk_logger("./logs_tgn", vars(self.args))
        
        for ep in range(self.args.epochs):
            loss, ap  = self.train_epoch()
            ap = torch.tensor([ap], device=self.ctx.device)
            dist.all_reduce(ap, op=dist.ReduceOp.SUM)
            ap = ap.item() / self.ctx.size
            self.ctx.sync_print(f"Epoch {ep+1} | Loss: {loss:.4f} | AP: {ap:.4f}")
            if ep % 10 == 0 and self.ctx.rank == 0:
                torch.save(self.model.state_dict(), f"./checkpoints/tgn_ep{ep}.pth")


if __name__ == "__main__":
    # 程序入口
    engine = TrainingEngine()
    engine.run()