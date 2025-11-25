from typing import List, Optional
import dgl
import starrygl
from starrygl import distributed
import os.path as osp
import torch
import torch.distributed as dist
from torch_geometric.data import Data
import starrygl
import starrygl.lib.libstarrygl_comm as starrygl_ops

class NegSamplingSet:
    def __init__(self, node_set):
        self.node_set = node_set
        
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
    
    # @staticmethod
    # def from_csrc_graph(csrc_graph) -> List[dgl.heterograph.]:
    #     if csrc_graph.layer_ptr.empty():
    #         return []
    #     return [dgl.heterograph({('node', 'edge', 'node'): (csrc_graph.src_nodes(), csrc_graph.dst_nodes())})]

    # @staticmethod
    # def from_csrc_list(csrc_graph_list) -> List[dgl.heterograph.]:
    #     graph_list = []
    #     for csrc_graph in csrc_graph_list:
    #         graph_list.append(pyGraph.from_csrc_graph(csrc_graph))
    #     return graph_list