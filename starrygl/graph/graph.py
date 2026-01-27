from typing import List, Optional
import dgl
import starrygl
from starrygl import distributed
import os.path as osp
import torch
import torch.distributed as dist
from torch_geometric.data import Data
import asyncio
import dgl
import starrygl
import starrygl.lib.libstarrygl_comm as starrygl_ops
from starrygl.utils.context import DistributedContext

class NegSamplingSet:
    def __init__(self, node_set):
        self.node_set = node_set
    
    def sample(self, num_samples: int, num_groups: int = 1)  -> torch.Tensor:
        """
        Sample a set of nodes for negative sampling.
        """
        if isinstance(self.node_set, torch.Tensor):
            return self.node_set[torch.randint(0, len(self.node_set), (num_samples, num_groups))]
        else:
            raise TypeError("node_set must be a torch.Tensor")



class pyGraph:
    def __init__(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        ts: torch.Tensor,
        node_num: int,
        chunk_size: int,
        eid: torch.Tensor = None,
        chunk_mapper: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        add_inverse_edge: bool = False,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        ctx = DistributedContext.get_default_context()
        print('size of src,dst,ts:', src.size(), dst.size(), ts.size(), node_num, chunk_size)
        if eid is None:
            eid = torch.arange(src.size(0), dtype=torch.long, device=ctx.device)
        self.add_inverse_edge = add_inverse_edge
        try:
            self.g = starrygl_ops.from_edge_index(node_num, chunk_size, 
                                              src.to(ctx.device), dst.to(ctx.device), ts.to(ctx.device), eid.to(ctx.device),
                                                  chunk_mapper,stream,ctx.local_rank)
        
        except Exception as e:
            print("Graph build failed:", e)
        print('finish build graph\n')
        
    def async_sample(
        self,
        time_start: int,
        time_end: int,
        chunk_list: torch.Tensor,  
        policy: dict = {},
        test_generate_kwargs: dict = {},
        op: Optional[str] = None,
    ):
        """
        Asynchronously sample a subgraph based on the given time range and chunk list.
        """
        if not isinstance(chunk_list, torch.Tensor):
            raise TypeError("chunk_list must be a torch.Tensor")
        
        if not isinstance(policy, dict):
            raise TypeError("policy must be a dict")
        
        if not isinstance(test_generate_kwargs, dict):
            raise TypeError("test_generate_kwargs must be a dict")
        
        if op is not None and not isinstance(op, str):
            raise TypeError("op must be a callable function")
        ctx = DistributedContext.get_default_context()
        test_generate_samples = test_generate_kwargs.get('test_generate_samples', torch.tensor([],dtype=torch.long,device = ctx.device))
        test_generate_samples_ts = test_generate_kwargs.get('test_generate_samples_ts', torch.tensor([],dtype=torch.long,device = ctx.device))
        if(policy['sample_type'] == 'cluster'):
            return self.g.submit_query(
                time_start, time_end, chunk_list.to(ctx.device), test_generate_samples.to(ctx.device), test_generate_samples_ts.to(ctx.device),
                'cluster', 1, 0, 0, 0, 0 ,0,op
            )
        else:
            if 'fanout' not in policy:
                raise ValueError("policy must contain 'fanout' for non-c sampling")
            if 'layers' not in policy:
                raise ValueError("policy must contain 'layers' for non-c sampling")
            
            #print('{} {} {} {} {}\n'.format(time_start, time_end, chunk_list, test_generate_samples, test_generate_samples_ts))
            #print(time_start, time_end, chunk_list, test_generate_samples, test_generate_samples_ts)
            return self.g.submit_query(
                time_start, time_end, chunk_list.to(ctx.device), test_generate_samples.to(ctx.device), test_generate_samples_ts.to(ctx.device),
                policy['sample_type'], policy['layers'], policy['fanout'], policy['allowed_offset'],
                policy['equal_root_time'], policy['keep_root_time'],
                op
            )
    
    def get_last_sample(self):
        """
        Get the last sampled subgraph.
        """
        g0 = self.g.get()
        #await asyncio.sleep(0)
        return g0
    
    def slice_by_chunk_ts(
        self,
        chunk_list: torch.Tensor,
        time_begin: float,
        time_end: float,
    ) -> 'pyGraph':
        sub_g = self.g.slice_by_chunk_ts(
            chunk_list, time_begin, time_end
        )
        
        return sub_g
    
    def sample_src_in_chunks_khop(
        self,
        chunk_list: torch.Tensor,
        k: int,
        layers: List[int],
        time_begin: float,
        time_end: float,
    ) -> Data:
        sampled_data = self.g.sample_src_in_chunks_khop(
            chunk_list, k, layers, time_begin, time_end
        )
        return sampled_data
        #return pyGraph.from_csrc_graph(sampled_data)
    

class AsyncBlock:
    def __init__(self, graph: pyGraph, index: int):
        self.graph = graph
        self.index = index
        
    def value(self):
        out = self.graph.get_last_sample()
        return out
        # nid_mapper = out.nodes_remapper_id
        # root = out.roots
        # neg_roots = out.neg_roots
        # neighbors = out.neighbors_list
        # return (root, neg_roots, neighbors, nid_mapper)
    
    @staticmethod
    def to_block(root, neg_roots, neighbors, nid_mapper, self_loop = True, add_inverse_edge = False):
        mfgs = []
        root_ts = root.ts
        nodes = root.roots
        last_layer_nodes = nodes
        last_layer_ts = root_ts
        for i in range(len(neighbors)):
            length = len(last_layer_nodes) if self_loop else 0
        
            start_ptr = neighbors[i].root_start_ptr
            end_ptr = neighbors[i].root_end_ptr

            if i == 0 and len(end_ptr) > 0:  
                start_ptr = start_ptr[:-1]
                end_ptr = end_ptr[:-1]
            
                device = torch.device('cuda:{}'.format(torch.distributed.get_rank()))
                indptr = torch.cumsum(end_ptr - start_ptr, dim=0).to(device)
            
                # 原始邻居索引 (Raw Indices)
                raw_indices_base = torch.arange(neighbors[i].start_ptr[-1], device=device)
            
                # 处理重复采样/重排 (C++层面的逻辑)
                start = torch.repeat_interleave(start_ptr, end_ptr - start_ptr, dim=0)
                new_ind = torch.arange(start_ptr.size(0), device=device)
                ind = new_ind - indptr + start
            
                # raw_indices: 对应 neighbors[i] 数据数组的真实下标
                raw_indices = raw_indices_base[ind]
            
                # graph_indices: 构建 DGL Block 用的下标 (做了偏移，因为前排留给了 self-loop)
                graph_indices = raw_indices + length
            
                edge_ids = torch.tensor([], dtype=torch.int64, device=DistributedContext.get_default_context().device)
            
                # 计算总源节点数: Self-loops (length) + Neighbors (max index + 1)
                # 必须转为 int
                n_neighbors = raw_indices.max().item() + 1 if raw_indices.numel() > 0 else 0
                num_src = length + n_neighbors
            
                b = dgl.create_block(
                    ('csc', (indptr, graph_indices, edge_ids)),
                    num_src_nodes=num_src,
                    num_dst_nodes=len(start_ptr)
                )
                
                if self_loop:
                    # [关键修正]
                    # 1. Self部分: 使用 last_layer_ts (即 root.ts)
                    # 2. Neighbor部分: 使用 raw_indices 从 neighbors[i] 取数据
                    b.srcdata['ts'] = torch.cat((
                        last_layer_ts,
                        neighbors[i].neighbors_ts[raw_indices]
                    ))
                    b.srcdata['nid'] = torch.cat((
                        last_layer_nodes,
                        neighbors[i].neighbors[raw_indices]
                    ))
                else:
                    b.srcdata['ts'] = neighbors[i].neighbors_ts[raw_indices]
                    b.srcdata['nid'] = neighbors[i].neighbors[raw_indices]
                    
                # eids 也用 raw_indices
                b.edata['eid'] = neighbors[i].neighbors_eid[raw_indices]//2 if add_inverse_edge else neighbors[i].neighbors_eid[raw_indices]
                b.edata['dt'] = neighbors[i].neighbors_dt[raw_indices]
                
            # -------------------------------------------------
            # Case 2: i > 0 (后续层)
            # -------------------------------------------------
            else:
                device = torch.device('cuda:{}'.format(torch.distributed.get_rank()))
                indptr = start_ptr.to(device)
                
                # 原始邻居索引 (假设是紧凑的 0...M)
                raw_indices = torch.arange(indptr[-1], device=device)
                
                # 图索引 (偏移 length)
                graph_indices = raw_indices + length
                
                edge_ids = torch.tensor([], dtype=torch.int64, device=DistributedContext.get_default_context().device)
                
                # Src = Dst(上一层的Src) + 新的Neighbors
                num_src = length + len(raw_indices)
                
                b = dgl.create_block(
                    ('csc', (indptr, graph_indices, edge_ids)),
                    num_src_nodes=num_src,
                    num_dst_nodes=len(indptr) - 1
                )
                
                if self_loop:
                    b.srcdata['ts'] = torch.cat((
                        last_layer_ts,  # 这里已经包含了上一层的所有节点信息
                        neighbors[i].neighbors_ts[raw_indices]
                    ))
                    b.srcdata['nid'] = torch.cat((
                        last_layer_nodes,
                        neighbors[i].neighbors[raw_indices]
                    ))
                else:
                    b.srcdata['ts'] = neighbors[i].neighbors_ts[raw_indices]
                    b.srcdata['nid'] = neighbors[i].neighbors[raw_indices]
                    
                b.edata['eid'] = neighbors[i].neighbors_eid[raw_indices]//2 if add_inverse_edge else neighbors[i].neighbors_eid[raw_indices]
                b.edata['dt'] = neighbors[i].neighbors_dt[raw_indices] if neighbors[i].neighbors_dt.ndim > 0 else neighbors[i].neighbors_dt

            # -------------------------------------------------
            # 更新状态供下一层使用
            # -------------------------------------------------
            # [关键修正] 下一层的 Dst 数量 = 当前层的 Src 数量
            # 不需要 +=，直接读取当前构建好的 Block 的源节点信息即可
            last_layer_nodes = b.srcdata['nid']
            last_layer_ts = b.srcdata['ts']
            
            mfgs.append(b)

        mfgs.reverse()
        return root, neg_roots, mfgs, nid_mapper
                

class AsyncGraphBlob:
    def __init__(self,  
                 graph: pyGraph, 
                 time_start: List[int],
                time_end: List[int],
                chunk_list: List[torch.Tensor],  
                policy: dict = {},
                test_generate_kwargs: dict = {},
        ):
        self.graph = []
        for i in range(len(time_start)):
            graph.async_sample(
                time_start[i],
                time_end[i],
                chunk_list[i],
                policy=policy,
                test_generate_kwargs=test_generate_kwargs,
                op=('f' if i > 0 else 'r')  # Only the first graph gets the operation
            )
            self.graph.append(AsyncBlock(graph,i))
            


