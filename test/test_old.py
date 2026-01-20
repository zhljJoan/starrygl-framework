

import os
import sys
from os.path import abspath, join, dirname
from typing import List
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
from starrygl.cache.replica_table import CSRReplicaTable, UVACSRReplicaTable
from starrygl.graph.graph import pyGraph
from starrygl.data.chunk_dataloader import AsyncPipelineLoader, GraphCollator, StarryGLDataset
from starrygl.route.route import DistRouteIndex, PartitionState, Route
from starrygl.utils import DistributedContext
from starrygl.utils.params import *
from starrygl.utils.parser import parse_chunk_decay
from starrygl.data import PartitionData, STGraphLoader, STGraphBlob
from starrygl.cache.NodeState import NodeState

import starrygl.lib.libstarrygl_comm
#ops.sample_neighbor

print(dir(starrygl.lib.libstarrygl_comm))
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

        self.load_dataset()
        #self.load_model()
        
    def load_dataset(self):
        self.data_root: Path = Path(self.args.data_root).expanduser().resolve()
        self.data_name: str = self.args.dataset
        
        path = self.data_root / "processed" / f"{self.data_name}" / f"{self.ctx.size:03d}_{self.args.chunk_size:03d}_{self.args.hot_ratio:0.1f}_{self.ctx.rank+1:03d}.pth"
        lookup_table = self.data_root / "processed" / f"{self.data_name}" / f"{self.ctx.size:03d}_{self.args.hot_ratio:0.1f}_lookup_table.pth"
        self.lookup_table = torch.load(lookup_table, map_location="cpu")
        partition_book = CSRReplicaTable(
            indptr=self.lookup_table['indptr'],
            indices=self.lookup_table['indices'],
            locs=self.lookup_table['locs'],
            device=self.ctx.device
        )
        self.ctx.sync_print(f"loading {path}...")
        data = PartitionData.load(path)
        self.partition_state = PartitionState(
            loc_ids = data.node_data['nid'],
            loc_eids = data.edge_data['eid'],
            is_shared = DistRouteIndex(data.node_data['NID'][0].data).is_shared,
            partition_book=partition_book,
            dist_nid_mapper= self.lookup_table['distNID'].data,
            dist_eid_mapper=self.lookup_table['distEID'].data
        )
        #if self.args.load_full_sample_graph:
        #    path = self.data_root / "ctdg" / f"{self.data_name}.pth"
        #    full_data = torch.load(path)
        self.chunk_index = data.pop_ndata('c')[0].item()
        data.register_to_chunk_cached_data(node_chunk_id = self.chunk_index)
        data.register_to_edge_chunk_cached_data(node_chunk_id = self.chunk_index[data.edge_src.data])
        #self.data_config = load_config()
        self.cfg = get_config(dataset = self.data_name, model = self.args.model)
        dim_hidden = self.cfg["train"]["hidden_dim"]
        nodeState = NodeState(
            node_nums = self.chunk_index.numel(),
            dim = dim_hidden,
            partition_state=self.partition_state
        )
        self.chunk_count = self.chunk_index.max().item() + 1
        self.ctx.sync_print(f"chunk_count = {self.chunk_count}")
        self.ctx.sync_print(f"local_nodes = {self.chunk_index.numel()}")

        # dataloader
        num_nodes = torch.tensor([self.chunk_index.numel()], dtype=torch.long, device="cpu").item()
        print('num_nodes:{} check: {} {}\n'.format(num_nodes, data.edge_src.data.max().item(), data.edge_dst.data.max().item()))
        
        ctx = DistributedContext.get_default_context()
        self.graph_stream = torch.cuda.Stream(device=ctx.device)
        if(self.args.add_inverse_edge):
            print('add inverse')
            eid = torch.arange(data.edge_src.data.size(0), dtype=torch.long, device=ctx.device)
            src = torch.cat([data.edge_src.data, data.edge_dst.data], dim=0)
            dst = torch.cat([data.edge_dst.data, data.edge_src.data], dim=0)
            ts = torch.cat([data.edge_data['ts'].data, data.edge_data['ts'].data], dim=0)
            eid = torch.cat([eid, eid], dim=0)
        else:
            src = data.edge_src.data
            dst = data.edge_dst.data
            ts = data.edge_data['ts'].data
            eid = torch.arange(data.edge_src.data.size(0), dtype=torch.long, device=ctx.device)
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
        print('finish initlize\n')

        self.dataset = StarryGLDataset(node_chunk_id = self.chunk_index, ts=data.edge_data['ts'].data.long(), rank=dist.get_rank(), world_size=dist.get_world_size())
        
        # split training data and test data
        
        train_ratio = 0.4
        self.train_end_ts = data.edge_data['ts'].data.max().item()*train_ratio
        train_end_ts = torch.tensor([self.train_end_ts],device = ctx.device)
        dist.all_reduce(train_end_ts, torch.distributed.ReduceOp.MAX)
        self.train_end_ts = train_end_ts.item()
        self.train_dataset = self.dataset[: self.train_end_ts]
        collator = GraphCollator(data, nodeState, device=ctx.device)
        self.train_loader = AsyncPipelineLoader(
            dataset=self.train_dataset,
            graph_engine=g,
            collator=collator, # 注入 Collator
            sampling_config=self.cfg["sample"],
            partition_data=data,
            mode = 'CTDG',
            prefetch_factor=1
        )
        #self.test_loader = 
        #self.eval_loader = 

    def load_model(self):
        model_cls = eval(self.args.model.upper())
        hidden_dim: int = CONFIG_DATASET_TO_DIM[self.data_name][0]
        
        self.snaps_count: int = self.args.snaps_count
        self.fulls_count: int = self.args.fulls_count

        self.model: TGCN | MPNN_LSTM = model_cls(self.feats_dim, hidden_dim, self.label_dim).to(self.ctx.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.model = nn.parallel.DistributedDataParallel(self.model)
    
    def loss_fn(self, preds: List[Tensor], targs: List[Tensor]):
        assert len(preds) == len(targs)
        loss_acc, loss_cnt = 0.0, 0
        for i, (x, y) in enumerate(zip(preds, targs)):
            loss_acc += F.mse_loss(x, y) * self.part_loss_scale
            loss_cnt += 1
        return loss_acc / loss_cnt
    
    def yield_labels(self, seqs: STGraphBlob):
        for g in seqs.values():
            if g.is_block:
                if 'y' in g.dstdata:
                    y = g.dstdata['y']
                else:
                    y = g.srcdata['y']
            else:
                y = g.ndata['y']

            if y.dim() == 1:
                y = y.view(-1, 1)
            yield y
    
    def generate_chunk_order(self):
        if self.args.chunk_order == "rand":
            return torch.randperm(self.chunk_count, device=self.ctx.device)
        else:
            raise ValueError(f"Unknown chunk order: {self.args.chunk_order}")

    def train_epoch(self):
        #self.model.train()

        loss_acc, loss_cnt = 0.0, 0
        for seqs in self.train_loader:


            #preds = self.model(seqs)
            #targs = list(self.yield_labels(seqs))
            #loss = self.loss_fn(preds, targs)

            #loss.backward()
            #self.optimizer.step()
            #self.optimizer.zero_grad()

            #loss = loss.detach()
            loss = 0
            loss_acc += loss
            loss_cnt += 1
        
        loss = loss_acc
        if isinstance(loss, Tensor):
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss.item() / loss_cnt
        return loss
    
    @torch.no_grad()
    def eval_epoch(self):
        self.model.eval()

        loss_acc, loss_cnt = 0.0, 0
        for i, seqs in enumerate(self.loader(w=1)):
            preds = self.model(seqs)

            if i < self.train_end:
                continue

            targs = list(self.yield_labels(seqs))
            loss = self.loss_fn(preds, targs)

            loss_acc += loss
            loss_cnt += 1
        
        loss = loss_acc
        if isinstance(loss, Tensor):
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss.item() / loss_cnt
        return loss
    
    def run(self):
        self.ctx.init_disk_logger("./logs_flare", vars(self.args))
        self.ctx.info(ep=0, train_loss=0.0, eval_loss=0.0)

        route_timer = Route.get_timer()
        for ep in self.ctx.trange_epoch(self.args.epochs):
            route_timer.reset()
            #chunk_order = self.generate_chunk_order()
            
            train_loss = self.train_epoch()
            eval_loss = 0
            
            #eval_loss = self.eval_epoch()
            comm_time = route_timer.get()
            self.ctx.update_epoch(f"train_loss=>{train_loss:.4f} eval_loss=>{eval_loss:.4f} comm_time=>{comm_time:.4f}s")
            self.ctx.info(ep=ep+1, train_loss=train_loss, eval_loss=eval_loss, comm_time=comm_time)

        
class FlareEngine(TrainingEngine):
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)
        parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES)
        parser.add_argument("--data-root", type=str, default="~/DATA/FlareGraph")
        parser.add_argument("--epochs", type=int, default=200)
        parser.add_argument("--learning-rate", "--lr", dest="lr", type=float, default=1e-3)
        parser.add_argument("--snaps-count", type=int, default=8)
        parser.add_argument("--fulls-count", type=int, default=2)
        parser.add_argument("--no-routes", action="store_true", default=False)
        parser.add_argument("--no-states", action="store_true", default=False)
        return parser.parse_args()
    def __init__(self, args=None):
        if args is None:
            args = self.parse_args()
        self.args = args

        self.ctx = DistributedContext.init("nccl")

        self.load_dataset()
        self.load_model()
        
    def load_dataset(self):
        self.data_root: Path = Path(self.args.data_root).expanduser().resolve()
        self.data_name: str = self.args.dataset

        path = self.data_root / "processed" / f"{self.data_name}" / f"{self.ctx.size:03d}_{self.ctx.rank+1:03d}.pth"
        self.ctx.sync_print(f"loading {path}...")
        data = PartitionData.load(path)

        # num feats and num labels
        self.feats_dim: int = data.node_data['x'].data.size(-1)
        y = data.node_data['y'].data
        if y.dtype == torch.long:
            self.label_dim = y.max().item() + 1 if y.dim() == 1 else y.size(-1)
        elif y.dim() == 1:
            self.label_dim = 1 if y.dim() == 1 else y.size(-1)
        else:
            raise ValueError(f"Unsupported label type: dim={y.dim()} dtype={y.dtype}")
        del y

        # temporal decay
        chunk_index = data.pop_ndata('c')[0].item()
        self.chunk_count = chunk_index.max().item() + 1
        self.ctx.sync_print(f"chunk_count = {self.chunk_count}")
        self.ctx.sync_print(f"local_nodes = {chunk_index.numel()}")

        self.chunk_decay = parse_chunk_decay(
            pattern=self.args.chunk_decay,
            chunk_count=self.chunk_count,
            snaps_count=self.args.snaps_count,
            fulls_count=self.args.fulls_count,
        )
        self.ctx.sync_print(f"chunk_decay = {self.chunk_decay}")

        # dataloader
        self.loader = STGraphLoader.from_partition_data(data=data, device=self.ctx.device, chunk_index=chunk_index)

        # split training data and test data
        train_ratio = 0.4
        self.train_end = int(len(self.loader) * train_ratio)
        self.train_loader = self.loader[:self.train_end]

        # compute partition loss scale
        num_nodes = torch.tensor([chunk_index.numel()], dtype=torch.long, device="cpu")
        dist.all_reduce(num_nodes, op=dist.ReduceOp.SUM, group=self.ctx.cpu_group)
        self.part_loss_scale = chunk_index.numel() / num_nodes.item()

    def load_model(self):
        model_cls = eval(self.args.model.upper())
        hidden_dim: int = CONFIG_DATASET_TO_DIM[self.data_name][0]
        
        self.snaps_count: int = self.args.snaps_count
        self.fulls_count: int = self.args.fulls_count

        self.model: TGCN | MPNN_LSTM = model_cls(self.feats_dim, hidden_dim, self.label_dim).to(self.ctx.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.model = nn.parallel.DistributedDataParallel(self.model)
    
    def loss_fn(self, preds: List[Tensor], targs: List[Tensor]):
        assert len(preds) == len(targs)
        loss_acc, loss_cnt = 0.0, 0
        for i, (x, y) in enumerate(zip(preds, targs)):
            loss_acc += F.mse_loss(x, y) * self.part_loss_scale
            loss_cnt += 1
        return loss_acc / loss_cnt
    
    def yield_labels(self, seqs: STGraphBlob):
        for g in seqs.values():
            if g.is_block:
                if 'y' in g.dstdata:
                    y = g.dstdata['y']
                else:
                    y = g.srcdata['y']
            else:
                y = g.ndata['y']

            if y.dim() == 1:
                y = y.view(-1, 1)
            yield y
    
    def generate_chunk_order(self):
        if self.args.chunk_order == "rand":
            return torch.randperm(self.chunk_count, device=self.ctx.device)
        else:
            raise ValueError(f"Unknown chunk order: {self.args.chunk_order}")

    def train_epoch(self, chunk_order: Tensor):
        self.model.train()

        loss_acc, loss_cnt = 0.0, 0
        for seqs in self.train_loader(
            chunk_order=chunk_order,
            chunk_decay=self.chunk_decay,
            w=self.fulls_count,
        ):
            seqs.erase_states_(self.args.no_states)
            seqs.erase_routes_(self.args.no_routes)

            preds = self.model(seqs)
            targs = list(self.yield_labels(seqs))
            loss = self.loss_fn(preds, targs)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss = loss.detach()

            loss_acc += loss
            loss_cnt += 1
        
        loss = loss_acc
        if isinstance(loss, Tensor):
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss.item() / loss_cnt
        return loss
    
    @torch.no_grad()
    def eval_epoch(self):
        self.model.eval()

        loss_acc, loss_cnt = 0.0, 0
        for i, seqs in enumerate(self.loader(w=1)):
            preds = self.model(seqs)

            if i < self.train_end:
                continue

            targs = list(self.yield_labels(seqs))
            loss = self.loss_fn(preds, targs)

            loss_acc += loss
            loss_cnt += 1
        
        loss = loss_acc
        if isinstance(loss, Tensor):
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss.item() / loss_cnt
        return loss
    
    def run(self):
        self.ctx.init_disk_logger("./logs_flare", vars(self.args))
        self.ctx.info(ep=0, train_loss=0.0, eval_loss=0.0)

        route_timer = Route.get_timer()
        for ep in self.ctx.trange_epoch(self.args.epochs):
            route_timer.reset()
            chunk_order = self.generate_chunk_order()
            
            train_loss = self.train_epoch(chunk_order=chunk_order)
            eval_loss = self.eval_epoch()

            comm_time = route_timer.get()
            self.ctx.update_epoch(f"train_loss=>{train_loss:.4f} eval_loss=>{eval_loss:.4f} comm_time=>{comm_time:.4f}s")
            self.ctx.info(ep=ep+1, train_loss=train_loss, eval_loss=eval_loss, comm_time=comm_time)


class CTDGEngine(TrainingEngine):
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)
        parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES)
        parser.add_argument("--data-root", type=str, default="~/DATA/FlareGraph")
        parser.add_argument("--epochs", type=int, default=200)
        parser.add_argument("--learning-rate", "--lr", dest="lr", type=float, default=1e-3)
        parser.add_argument("--chunk-decay", type=str, default="auto:0.1")
        parser.add_argument("--chunk-order", type=str, default="rand", choices=["rand", "loss", "perb"])
        parser.add_argument("--fulls-count", type=int, default=2)
        parser.add_argument("--no-routes", action="store_true", default=False)
        parser.add_argument("--no-states", action="store_true", default=False)
        return parser.parse_args()
    def __init__(self, args=None):
        if args is None:
            args = self.parse_args()
        self.args = args

        self.ctx = DistributedContext.init("nccl")

        self.load_dataset()
        self.load_model()
        
    def load_dataset(self):
        self.data_root: Path = Path(self.args.data_root).expanduser().resolve()
        self.data_name: str = self.args.dataset
        
        path = self.data_root / "processed" / f"{self.data_name}" / f"{self.ctx.size:03d}_{self.ctx.rank+1:03d}.pth"
        self.ctx.sync_print(f"loading {path}...")
        data = PartitionData.load(path)
        

        # num feats and num labels
        self.feats_dim: int = data.node_data['f'].data.size(-1)
        

        # temporal decay
        chunk_index = data.pop_ndata('c')[0].item()
        self.chunk_count = chunk_index.max().item() + 1
        self.ctx.sync_print(f"chunk_count = {self.chunk_count}")
        self.ctx.sync_print(f"local_nodes = {chunk_index.numel()}")

        self.chunk_decay = parse_chunk_decay(
            pattern=self.args.chunk_decay,
            chunk_count=self.chunk_count,
            snaps_count=self.args.snaps_count,
            fulls_count=self.args.fulls_count,
        )
        self.ctx.sync_print(f"chunk_decay = {self.chunk_decay}")

        # dataloader
        self.loader = STGraphLoader.from_partition_data(data=data, device=self.ctx.device, chunk_index=chunk_index)

        # split training data and test data
        train_ratio = 0.4
        self.train_end = int(len(self.loader) * train_ratio)
        self.train_loader = self.loader[: self.train_end]

        # compute partition loss scale
        num_nodes = torch.tensor([chunk_index.numel()], dtype=torch.long, device="cpu")
        dist.all_reduce(num_nodes, op=dist.ReduceOp.SUM, group=self.ctx.cpu_group)
        self.part_loss_scale = chunk_index.numel() / num_nodes.item()

    def load_model(self):
        model_cls = eval(self.args.model.upper())
        hidden_dim: int = CONFIG_DATASET_TO_DIM[self.data_name][0]
        
        self.snaps_count: int = self.args.snaps_count
        self.fulls_count: int = self.args.fulls_count

        self.model: TGCN | MPNN_LSTM = model_cls(self.feats_dim, hidden_dim, self.label_dim).to(self.ctx.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.model = nn.parallel.DistributedDataParallel(self.model)
    
    def loss_fn(self, preds: List[Tensor], targs: List[Tensor]):
        assert len(preds) == len(targs)
        loss_acc, loss_cnt = 0.0, 0
        for i, (x, y) in enumerate(zip(preds, targs)):
            loss_acc += F.mse_loss(x, y) * self.part_loss_scale
            loss_cnt += 1
        return loss_acc / loss_cnt
    
    def yield_labels(self, seqs: STGraphBlob):
        for g in seqs.values():
            if g.is_block:
                if 'y' in g.dstdata:
                    y = g.dstdata['y']
                else:
                    y = g.srcdata['y']
            else:
                y = g.ndata['y']

            if y.dim() == 1:
                y = y.view(-1, 1)
            yield y
    
    def generate_chunk_order(self):
        if self.args.chunk_order == "rand":
            return torch.randperm(self.chunk_count, device=self.ctx.device)
        else:
            raise ValueError(f"Unknown chunk order: {self.args.chunk_order}")

    def train_epoch(self, chunk_order: Tensor):
        self.model.train()

        loss_acc, loss_cnt = 0.0, 0
        for seqs in self.train_loader(
            chunk_order=chunk_order,
            chunk_decay=self.chunk_decay,
            w=self.fulls_count,
        ):
            seqs.erase_states_(self.args.no_states)
            seqs.erase_routes_(self.args.no_routes)

            preds = self.model(seqs)
            targs = list(self.yield_labels(seqs))
            loss = self.loss_fn(preds, targs)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss = loss.detach()

            loss_acc += loss
            loss_cnt += 1
        
        loss = loss_acc
        if isinstance(loss, Tensor):
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss.item() / loss_cnt
        return loss
    
    @torch.no_grad()
    def eval_epoch(self):
        self.model.eval()

        loss_acc, loss_cnt = 0.0, 0
        for i, seqs in enumerate(self.loader(w=1)):
            preds = self.model(seqs)

            if i < self.train_end:
                continue

            targs = list(self.yield_labels(seqs))
            loss = self.loss_fn(preds, targs)

            loss_acc += loss
            loss_cnt += 1
        
        loss = loss_acc
        if isinstance(loss, Tensor):
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss.item() / loss_cnt
        return loss
    
    def run(self):
        self.ctx.init_disk_logger("./logs_flare", vars(self.args))
        self.ctx.info(ep=0, train_loss=0.0, eval_loss=0.0)

        route_timer = Route.get_timer()
        for ep in self.ctx.trange_epoch(self.args.epochs):
            route_timer.reset()
            chunk_order = self.generate_chunk_order()
            
            train_loss = self.train_epoch(chunk_order=chunk_order)
            eval_loss = self.eval_epoch()

            comm_time = route_timer.get()
            self.ctx.update_epoch(f"train_loss=>{train_loss:.4f} eval_loss=>{eval_loss:.4f} comm_time=>{comm_time:.4f}s")
            self.ctx.info(ep=ep+1, train_loss=train_loss, eval_loss=eval_loss, comm_time=comm_time)

if __name__ == "__main__":
    TrainingEngine().run()