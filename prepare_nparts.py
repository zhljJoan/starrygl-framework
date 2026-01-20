import os
import torch

from torch import Tensor
from typing import *

from pathlib import Path
from tqdm import trange, tqdm

import dgl


def get_node_types(g: dgl.DGLGraph, bins: int = 10) -> Tensor:
    w: Tensor = g.in_degrees() + g.out_degrees()
    w = w.float()

    b = torch.histogram(w, bins=bins).bin_edges
    ntypes = torch.bucketize(w, b[1:])

    c = 0
    for i in range(bins):
        x = torch.count_nonzero(ntypes == i).item()
        if x > 0:
            ntypes[ntypes == i] = c
            c += 1
    print(f"{ntypes.max().item() + 1}")
    return ntypes

def apply_inter_partition(g: dgl.DGLGraph, node_types: Tensor | None, k: int) -> Tensor:
    assert k < 100, f"k must be less than 100"
    print("Enter apply_inter_partition()")
    
    node_parts: Tensor = metis(g, k=k, balance_ntypes=node_types, balance_edges=True)
    assert node_parts.max().item() + 1 == k, f"node_parts.max().item() + 1 != k"
    return node_parts.type(torch.uint8)

def apply_intra_partition(g: dgl.DGLGraph, node_parts: Tensor, is_shared:Tensor, k: int) -> Tensor:
    assert k < 1000, f"k must be less than 1000"
    print("Enter apply_intra_partition()")

    num_parts = node_parts.max().item() + 1
    out = node_parts.type(torch.int32) << 16 
    print(f"num_parts max: {node_parts.max().item()+1}")
    for i in range(num_parts):
        print(f"Enter apply_intra_partition():{i}/{num_parts}")
        p = dgl.node_subgraph(g, (node_parts == i)) #| (is_shared == 1))
        print(f"subgraph num nodes: {p.num_nodes(), p.num_edges()}")
        x = metis(p, k=k, balance_edges=True)
        
        print('metis is {}'.format(x.type(torch.int32)))
        out[p.ndata[dgl.NID]] |= x.type(torch.int32) & 0xFFFF
    print('max sum is {}'.format((out >> 30).max().item()))
    return out


if __name__ == "__main__":
    # src_root = "~/DATA/starrygl/web"
    # tgt_root = "~/DATA/starrygl/nparts"
    src_root = "/mnt/data/zlj/starrygl-data/ctdg"
    tgt_root = "/mnt/data/zlj/starrygl-data/nparts"
    num_inter_parts = 4
    num_intra_parts = 128
    hot_nodes_ratio = 0.1
    src_root = Path(src_root).expanduser().resolve()
    tgt_root = Path(tgt_root).expanduser().resolve()

    tgt_root.mkdir(parents=True, exist_ok=True)

    large_data_names = ["WIKI"]
    large_data_flags = False

    metis = dgl.distributed.partition.metis_partition_assignment
    print(src_root)
    for p in src_root.glob("*.pth"):

        name = p.stem
        if os.path.exists(tgt_root / f"{name}_{num_inter_parts:03d}_{hot_nodes_ratio:0.1f}.pth"):
            print(f"skip {name}...")
            continue
        
        if name == 'BCB' or name == 'soc-flickr-growth':
            continue
        if large_data_flags:
            if name not in large_data_names:
                continue


        print(f"loading {name}...")
        state_dict = torch.load(str(p), mmap=True)
        
        num_nodes: int = state_dict["num_nodes"]
        print(f"num nodes: {num_nodes}")
        dataset: List[Dict[str, Tensor]] = state_dict["dataset"]
        if state_dict["num_snapshots"] == 0:
            edge_index = dataset["edge_index"][:2,:]
            edge_ts = dataset["edge_index"][2]
        else:
            edge_index = torch.cat([data["edge_index"] for data in dataset], dim=1)
            
            
        print(f"building graph {name}...")
        src = edge_index[0]
        dst = edge_index[1]

        g = dgl.graph((src, dst), num_nodes=num_nodes, idtype=torch.int64)
        # ntypes = get_node_types(g, 3)
        ntypes = None
        if hot_nodes_ratio > 0:
            if isinstance(dataset,List):
                deg = sum([dataset[i]['deg'] for i in range(len(dataset))])
            elif 'deg' in dataset:
                deg = dataset["deg"]
            num_hot = int(num_nodes * hot_nodes_ratio)
            _, idx = torch.topk(deg, num_hot)
            is_hot = torch.zeros(num_nodes, dtype=torch.bool)
            is_hot[idx] = 1
            print(f"hot nodes: {is_hot.sum().item()}")
        else:
            is_hot = torch.zeros(num_nodes, dtype=torch.bool)
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.to_bidirected(g)
        g0 = dgl.remove_edges(g, idx)
        print(f"partitioning {name} with k={num_inter_parts}, topk={hot_nodes_ratio}...")
        inter_nparts = apply_inter_partition(g0, node_types=ntypes, k=num_inter_parts)
        #intra_nparts = apply_intra_partition(g, inter_nparts, is_shared=is_hot, k=num_intra_parts)
        #if hot_nodes_ratio > 0:
        #    inter_nparts |= (is_hot.type(torch.int32) << 14)
            #intra_nparts |= (is_hot.type(torch.int32) << 30)
        #print(hot_nodes_ratio, (intra_nparts >> 30).sum().item())

        inter_p = tgt_root / f"{name}_{num_inter_parts:03d}_{hot_nodes_ratio:0.1f}.pth"
        intra_p = tgt_root / f"{name}_{num_inter_parts:03d}_{num_intra_parts:04d}_{hot_nodes_ratio:0.1f}.pth"

        print(f"{name}: {num_nodes}")
        for i in range(num_inter_parts):
            m = inter_nparts == i
            num_part_nodes = torch.count_nonzero(m).item()
            #x = intra_nparts[m] & 0xFFFF
            #x = [torch.count_nonzero(x == j).item() for j in range(num_intra_parts)]
            #x = torch.tensor(x).float()
            #std, mean = torch.std_mean(x)
            #print(f"  part {i}: {num_part_nodes} std={std.item():.2f} mean={mean.item():.2f}")

        
        print(f"saving {inter_p}...")
        torch.save((inter_nparts,is_hot), str(inter_p))

        #print(f"saving {intra_p}...")
        #torch.save(intra_nparts, str(intra_p))
