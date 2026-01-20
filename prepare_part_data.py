import torch

from torch import Tensor
from typing import *

import dgl

from pathlib import Path
from tqdm import tqdm

from torch_scatter import scatter_add

from starrygl.route import *
from starrygl.route.route import Route
from starrygl.data import *


if __name__ == "__main__":
    src_root = "/mnt/data/zlj/starrygl-data/web"
    tgt_root = "/mnt/data/zlj/starrygl-data/"
    #src_root = "~/DATA/FlareGraph/web"
    #tgt_root = "~/DATA/FlareGraph"

    src_root = Path(src_root).expanduser().resolve()
    tgt_root = Path(tgt_root).expanduser().resolve()

    large_data_names = ["WIKI"]
    large_data_flags = False

    """
    key = num of partition
    val = (num of parts, num of chunks per part)
    """
    params = {
        # 1: (4, 128),
        4: (4, 128, 0.1),
        # 8: (8, 64),
        # 12: (12, 48),
        # 16: (16, 32),
    }

    for p in src_root.glob("*.pth"):
        name = p.stem

        if large_data_flags:
            if name not in large_data_names:
                continue


        print(f"loading {name}...")
        state_dict = torch.load(str(p), mmap=True)

        num_nodes: int = state_dict["num_nodes"]
        dataset: List[Dict[str, Tensor]] = state_dict["dataset"]

        data_root = tgt_root / "processed" / name
        data_root.mkdir(parents=True, exist_ok=True)

        for k in params:
            kk1, kk2, kk3 = params[k]
            node_parts, is_shared = torch.load(tgt_root / "nparts" / f"{name}_{kk1:03d}_{kk3:0.1f}.pth").long()
            
            #chunk_index = node_parts & 0xFFFF
            #chunk_index = chunk_index.to(torch.int16)

            node_parts = (node_parts >> 16 & 0x7FFF)
            is_shared = (node_parts & 0x8000)
            print('is shared {}'.format(is_shared.sum().item()))
            pdata = [[] for _ in range(k)]
            for data in tqdm(dataset, desc=f"name={name}, k={k}"):
                if data.get("y", None) is None:
                    continue
                edge_index = data["edge_index"].long()
                edge_weight = data["edge_weight"].float()

                in_deg = scatter_add(edge_weight, edge_index[1], dim=0, dim_size=num_nodes)
                out_deg = scatter_add(edge_weight, edge_index[0], dim=0, dim_size=num_nodes)

                in_deg = torch.sqrt(in_deg)[edge_index[1]]
                out_deg = torch.sqrt(out_deg)[edge_index[0]]
                gcn_norm = edge_weight / (in_deg * out_deg)
                gcn_norm = gcn_norm.nan_to_num_(0.0)
                assert not gcn_norm.isinf().any()
                
                gs = Route.from_graph(node_parts, edge_index, num_parts=k)

                for i, g in enumerate(gs):
                    src_ids = g.srcdata[dgl.NID].long()
                    dst_ids = g.dstdata[dgl.NID].long()
                    edge_ids = g.edata[dgl.EID].long()
                    
                    g.srcdata["x"] = data["x"][src_ids]
                    g.dstdata["y"] = data["y"][dst_ids]
                    g.dstdata["c"] = chunk_index[dst_ids]

                    g.edata["gcn_norm"] = gcn_norm[edge_ids]
                    g.edata["w"] = edge_weight[edge_ids]

                    pdata[i].append(g)
            
            for i, gs in enumerate(tqdm(pdata, desc=f"saving...")):
                path = data_root / f"{k:03d}_{i+1:03d}.pth"
                data = PartitionData.from_blocks(gs)
                data.save(path)
                data.load(path)
