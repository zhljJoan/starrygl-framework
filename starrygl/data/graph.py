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
        chunk_mapper: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        print('size of src,dst,ts:', src.size(), dst.size(), ts.size(), node_num, chunk_size)
        try:
            self.g = starrygl_ops.from_edge_index(node_num, chunk_size, 
                                              src.to('cuda'), dst.to('cuda'), ts.to('cuda'), 
                                                  chunk_mapper,stream)
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
        
        if op is not None and not callable(op):
            raise TypeError("op must be a callable function")
        
        test_generate_samples = test_generate_kwargs.get('test_generate_samples', torch.tensor([]))
        test_generate_samples_ts = test_generate_kwargs.get('test_generate_samples_ts', torch.tensor([]))
        if(policy['sample_type'] == 'c'):
            return starrygl_ops.submit_query(
                time_start, time_end, chunk_list, test_generate_samples, test_generate_samples_ts,
                'c', 1, 0, 0, 0, 0 ,0,op
            )
        else:
            if 'num_layers' not in policy:
                raise ValueError("policy must contain 'num_layers' for non-c sampling")
            if 'k' not in policy:
                raise ValueError("policy must contain 'k' for non-c sampling")
            if 'layers' not in policy:
                raise ValueError("policy must contain 'layers' for non-c sampling")
            
            return starrygl_ops.submit_query(
                self.g, time_start, time_end, chunk_list, policy, test_generate_kwargs, 
                policy['sample_type'], policy['layers'], policy['fanout'], policy['allowed_offset'],
                policy['equal_root_time'], policy['keep_root_time'],
                op
            )
    
    def get_last_sample(self):
        """
        Get the last sampled subgraph.
        """
        g0 = starrygl_ops.get()
        #await asyncio.sleep(0)
        return g0
    
    def slice_by_chunk_ts(
        self,
        chunk_list: torch.Tensor,
        time_begin: float,
        time_end: float,
    ) -> 'pyGraph':
        sub_g = starrygl_ops.slice_by_chunk_ts(
            self.g, chunk_list, time_begin, time_end
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
        sampled_data = starrygl_ops.sample_src_in_chunks_khop(
            self.g, chunk_list, k, layers, time_begin, time_end
        )
        return sampled_data
        #return pyGraph.from_csrc_graph(sampled_data)
    

class AsyncBlock:
    def __init__(self, graph: pyGraph, index: int):
        self.graph = graph
        self.index = index
    def value(self):
        out = self.graph.get_last_sample()
        nid_mapper = out.nodes_remapper_id
        root = out.roots
        neg_roots = out.neg_roots
        neighbors = out.neighbors
        return (root, neg_roots, neighbors, nid_mapper)
        
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

