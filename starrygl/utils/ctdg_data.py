import torch

import numpy as np
import pandas as pd

from torch import Tensor
from pathlib import Path
from typing import *

from torch_scatter import scatter_add
import os


class CTDGDataLoader:
    def __init__(self,
        name: str,
        #window = 200,
        #skiprows = 2,
        #sep = '\s+',
        #skiptime = 1,
        #lags = 30,
        rand_dn = 172,
        rand_de = 172
    ) -> None:
        self.name = name
        self.rand_dn = rand_dn
        self.rand_de = rand_de

    

    def _read_web_data(self, path ):
        path = Path(path).expanduser().resolve()
        if path.suffix == ".edges":
            data = pd.read_csv(path, sep='\s+', header=None, skiprows=2).to_numpy()
            src = data[:, 0].astype(np.int64)
            dst = data[:, 1].astype(np.int64)
            if data.shape[1] == 3:
                time = data[:, 2].astype(np.int64)
            elif data.shape[1] == 4:
                time = data[:, 3].astype(np.int64)
        else:        
            data = pd.read_csv(path)
            src = np.array(data.src.values).astype(np.int64)
            dst = np.array(data.dst.values).astype(np.int64)
            time = np.array(data.time.values).astype(np.int64)
        num_nodes  = int(max(src.max().item(),dst.max().item())+1 )
        num_edges = int(src.shape[0])
        edges = torch.from_numpy(np.stack([src, dst, time ]))
        if path.suffix == ".edges":
            train_mask = torch.arange(num_edges) < int(num_edges * 0.7)
            val_mask = (torch.arange(num_edges) < int(num_edges * 0.85))& (torch.arange(num_edges) >= int(num_edges * 0.7))
            test_mask = torch.arange(num_edges) >= int(num_edges * 0.85)
        else:
            ext_roll_values = np.array(data.ext_roll.values).astype(np.int64)  
            train_mask = torch.from_numpy(ext_roll_values == 0)
            val_mask = torch.from_numpy(ext_roll_values == 1)
            test_mask = torch.from_numpy(ext_roll_values == 2)

        return edges, num_nodes, num_edges, train_mask, val_mask, test_mask
    

    def _read_feat(self, root, num_nodes, num_edges):
        path = root/'node_features.pt'
        node_feats = None
        if os.path.exists(path):
            node_feats = torch.load(path)
            if node_feats.dtype == torch.bool:
                node_feats = node_feats.type(torch.float32)
        if node_feats is None:
            node_feats = torch.randn(num_nodes,self.rand_dn)
        # else:
        #     if node_feats.shape[1] < self.rand_dn:
        #         _node_feats = torch.zeros(num_nodes,self.rand_dn)
        #         _node_feats[:,:node_feats.shape[1]] = node_feats
        #         node_feats = _node_feats
        edge_feats = None
        path = root/'edge_features.pt'
        if os.path.exists(path):
            edge_feats = torch.load(path)
            if edge_feats.dtype == torch.bool:
                edge_feats = edge_feats.type(torch.float32)
        if edge_feats is None :
            edge_feats = torch.randn(num_edges, self.rand_de)
        return node_feats,edge_feats
    
    def get_dataset(self, root = None):
        root = Path(root).expanduser().resolve()
        path = root/ self.name / "edges.csv"
        if not path.exists():
            path = root / self.name / f"{self.name}.edges"
            if not path.exists():
                print(f"File not found: {path}")
                return None
        edges, num_nodes, num_edges, train_mask, val_mask, test_mask = self._read_web_data(path)
        i_deg = scatter_add(torch.ones(num_edges),edges[1],dim=0, dim_size = num_nodes).int()
        o_deg = scatter_add(torch.ones(num_edges),edges[0],dim=0, dim_size = num_nodes).int()
        node_feats,edge_feats = self._read_feat(root, num_nodes = num_nodes, num_edges = num_edges)
        dataset = {
            "edge_index": edges,
            "node_feat": node_feats,
            "edge_feat": edge_feats,
            "deg": i_deg + o_deg,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask
        }
        state_dict = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_snapshots": 0,
            "dataset": dataset
        }
        return state_dict





if __name__ == "__main__":
    #src_root = "~/DATA/DynaHB"
    #tgt_root = "~/DATA/FlareGraph"
    src_root = "/mnt/nfs/zlj/TGL-DATA"
    #src_root = "/mnt/data/zlj/tgl_data/DATA"
    #src_root = "/mnt/data/zlj/starrygl-data/raw"
    tgt_root = "/mnt/data/zlj/starrygl-data"
    root = Path(tgt_root).expanduser().resolve() / "ctdg"
    root.mkdir(parents=True, exist_ok=True)
    for p in Path(src_root).expanduser().resolve().glob("*/"):
        name = p.stem
        if name not in ['StackOverflow','WikiTalk']:
            continue
        #if name in ['Flights','MOOC','LASTFM','REDDIT','stackoverflow','WIKI', 'wikitalk']:
        #    continue
        print(f"Processing {name}...")
        dataloader = CTDGDataLoader(name)
        data = dataloader.get_dataset(src_root)
        if data is None:
            print(f"Failed to load {name}")
            continue
        torch.save(data, root/f"{dataloader.name}.pth")
        num_nodes: int = data["num_nodes"]
        num_edges: int = data["num_edges"]
        print(dataloader.name, num_nodes, num_edges)