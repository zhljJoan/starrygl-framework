import torch

from torch import Tensor
from typing import *

import dgl

from pathlib import Path
from tqdm import tqdm

from torch_scatter import scatter_add

from starrygl.core import *
from starrygl.core.route import DistRouteIndex, Route
from starrygl.data import *


if __name__ == "__main__":
    src_root = "/mnt/data/zlj/starrygl-data/ctdg"
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
    batch_size = 1000

    for p in src_root.glob("*.pth"):
        name = p.stem

        if large_data_flags:
            if name not in large_data_names:
                continue


        print(f"loading {name}...")
        state_dict = torch.load(str(p), mmap=True)

        num_nodes: int = state_dict["num_nodes"]
        dataset: Dict[str, Tensor] = state_dict["dataset"]

        data_root = tgt_root / "processed" / name
        data_root.mkdir(parents=True, exist_ok=True)

        for k in params:
            kk1, kk2, kk3 = params[k]
            node_parts = torch.load(tgt_root / "nparts" / f"{name}_{kk1:03d}_{kk2:04d}_{kk3:0.1f}.pth").long()
            
            chunk_index = node_parts & 0xFFFF
            chunk_index = chunk_index.to(torch.int32)

            node_parts = (node_parts >> 16 & 0x3FFF)
            is_shared = (node_parts >> 14).bool()
            print(is_shared.sum(), node_parts.max())
            chunk_index[is_shared] = 0xFFFF
            pdata = []
            data = state_dict["dataset"]
            edge_index = data['edge_index'].long()
            src,dst,ts= edge_index
            g = dgl.create_block((torch.cat((src,dst)),torch.cat((dst,src))), num_src_nodes = num_nodes, num_dst_nodes = num_nodes)
            part_mat = torch.zeros(num_nodes, k, dtype = torch.float)
            part_nodes = [is_shared.nonzero().squeeze() for _ in range(k)]
            part_nodes_dist_id = []
            num_shared_nodes = len(part_nodes)
            xmp = torch.zeros(k,num_nodes,dtype=torch.int)
            
            for i in range(k): 
                part_mat[:,i] = ((node_parts == i) | is_shared).float()
                part_nodes[i] = (torch.cat((part_nodes[i],torch.nonzero((node_parts == i) & ~is_shared).squeeze())))  
                print(len(part_nodes[i]), is_shared.shape, torch.cat((src,dst)).unique().shape)
                part_nodes_dist_id.append(DistRouteIndex(part_nodes[i],torch.full((len(part_nodes[i]),),i,dtype=torch.int16)))
                part_nodes_dist_id[i].set_shared(slice(0,num_shared_nodes))
                xmp[i,part_nodes[i]] = torch.arange(0,len(part_nodes[i]),dtype=torch.int)
            #按照边所在邻居位置分配边的分区
            g.srcdata['p'] = part_mat
            in_neighbors = dgl.ops.copy_u_sum(g,part_mat)
            _,edge_part = torch.max(in_neighbors[src]+in_neighbors[dst],dim = 1)
            edge_part[is_shared[src] & ~is_shared[dst]] = node_parts[dst[is_shared[src] & ~is_shared[dst]]]
            edge_part[is_shared[dst] & ~is_shared[src]] = node_parts[src[is_shared[dst] & ~is_shared[src]]]
            #按照dst的位置分配邻居分区
            #pass
            #按照src的位置分配邻居分区
            #pass
            origin_eid = torch.arange(edge_index.size(1),dtype=torch.long)
            origin_nid = torch.arange(num_nodes,dtype=torch.long)
            part_path = data_root / f"partlist.pth"
            torch.save((node_parts,edge_part),part_path)
            
            for i in range(k):
                k_src = edge_index[0,edge_part==i]
                outer_source_nodes = k_src[(~is_shared[k_src])|(node_parts[k_src]!=i)].unique()
                outer_source_nodes_part = node_parts[outer_source_nodes]
                print(k_src.shape,outer_source_nodes.shape)
                print(outer_source_nodes_part.max())
                outer_source_nodes_rid = DistRouteIndex(xmp[outer_source_nodes_part,outer_source_nodes], outer_source_nodes_part)
                xmp[i,outer_source_nodes] = torch.arange(len(part_nodes[i]),len(part_nodes[i])+len(outer_source_nodes),dtype=torch.int)
                
                k_dst = edge_index[1,edge_part==i]
                outer_dst_nodes = k_dst[((~is_shared[k_dst])|(node_parts[k_dst]!=i))]
                outer_dst_nodes = outer_dst_nodes[xmp[i,outer_dst_nodes] == 0].unique()                
                print(k_dst.shape,outer_dst_nodes.shape  )
                outer_dst_nodes_part = node_parts[outer_dst_nodes]
                outer_dst_nodes_rid = DistRouteIndex(xmp[outer_dst_nodes_part,outer_dst_nodes], outer_dst_nodes_part)
                xmp[i,outer_dst_nodes] = torch.arange(len(part_nodes[i])+len(outer_source_nodes),len(part_nodes[i])+len(outer_source_nodes)+len(outer_dst_nodes),dtype=torch.int)
                
                num_src_nodes = len(part_nodes[i]) + len(outer_source_nodes)
                num_dst_nodes = len(part_nodes[i]) + len(outer_source_nodes) + len(outer_dst_nodes)         
                print(num_src_nodes,num_dst_nodes)
                g = dgl.create_block((xmp[i,k_src],xmp[i,k_dst]), num_src_nodes = num_src_nodes, num_dst_nodes = num_dst_nodes,idtype=torch.int64)
                g.srcdata[dgl.NID] = torch.arange(num_src_nodes,dtype=torch.long)
                g.dstdata[dgl.NID] = torch.arange(num_dst_nodes,dtype=torch.long)
                g.edata[dgl.EID] = torch.arange(g.num_edges(),dtype = torch.long)
                g.dstdata['NID'] = torch.cat([part_nodes_dist_id[i].dist,outer_source_nodes_rid.dist,outer_dst_nodes_rid.dist])
                g.dstdata['nid'] = origin_nid[torch.cat((part_nodes[i],outer_source_nodes,outer_dst_nodes))]
                
                #print(g.dstdata['N'].keys())
                part_nodes[i] = torch.cat([part_nodes[i],outer_source_nodes,outer_dst_nodes])
                g.dstdata['c'] = chunk_index[part_nodes[i]]
                g.edata['eid'] = origin_eid[edge_part == i]
                g.edata['EID'] = DistRouteIndex(torch.arange(g.num_edges()),torch.full((g.num_edges(),),i,dtype=torch.int16)).dist
                #print(part_nodes[i].shape, g.num_nodes())
                g.dstdata['f'] = data['node_feat'][part_nodes[i]] 
                #print(g.ndata['_N']['f'])
                g.edata['f'] = data['edge_feat'][edge_part == i]
                g.edata['train_mask'] = data['train_mask'][edge_part == i]
                g.edata['val_mask'] = data['val_mask'][edge_part == i]
                g.edata['test_mask'] = data['test_mask'][edge_part == i]
                g.edata['ts'] = ts[edge_part == i]
                #如果是动态特征
                #g.srcdata[] = 
                g.route = None
                pdata.append(g)
            
            
            for i, gs in enumerate(tqdm(pdata, desc=f"saving...")):
                path = data_root / f"{k:03d}_{i+1:03d}.pth"
                #print(gs.dstdata['_N'].keys())
                data = PartitionData.from_blocks(gs)
                data.save(path)
                #torch.save(gs,path)

                
