import random
import sys
from os.path import abspath, join, dirname

from starrygl.distributed.context import DistributedContext

sys.path.insert(0, join(abspath(dirname(__file__))))
from torch import Tensor
import torch
from base import NegativeSampling
from base import NegativeSamplingMode
from typing import Any, List, Optional, Tuple, Union

class LocalNegativeSampling(NegativeSampling):

    def __init__(
        self,
        mode: str,
        amount: Union[int, float] = 1,
        unique: bool = False,
        src_node_list: torch.Tensor = None,
        dst_node_list: torch.Tensor = None,
        chunk_index: Optional[int] = None,
    ):
        super(LocalNegativeSampling,self).__init__(mode,amount,unique=unique)
        self.src_node_list = src_node_list.to('cpu') if src_node_list is not None else None
        self.dst_node_list = dst_node_list.to('cpu') if dst_node_list is not None else None
        self.rdm = torch.Generator()
        if seed is not None:
            random.seed(seed)
        seed = random.randint(0,100000)
        print('seed is',seed)
        ctx = DistributedContext.get_default_context()
        self.rdm.manual_seed(seed^ctx.rank)
        self.rdm = torch.Generator()
        self.local_mask = local_mask
        if self.local_mask is not None:
            self.local_dst = dst_node_list[local_mask]
        self.prob = prob
        #self.rdm.manual_seed(42)
        #print('dst_nde_list {}\n'.format(dst_node_list))
    def is_binary(self) -> bool:
        return self.mode == NegativeSamplingMode.binary

    def is_triplet(self) -> bool:
        return self.mode == NegativeSamplingMode.triplet

    def sample(self, num_samples: int,
               num_nodes: Optional[int] = None) -> Tensor:
        r"""Generates :obj:`num_samples` negative samples."""
        if self.is_binary():
            if self.src_node_list is None or self.dst_node_list is None:
                return torch.randint(num_nodes, (num_samples, )),torch.randint(num_nodes, (num_samples, ))
            else:
                self.src_node_list[torch.randint(len(self.src_node_list), (num_samples, ))],
                self.dst_node_list[torch.randint(len(self.dst_node_list), (num_samples, ))]
        else:   
            if self.dst_node_list is None:
                return torch.randint(num_nodes, (num_samples, ),generator=self.rdm)
            elif self.local_mask is not None:
                p = torch.rand(size=(num_samples,))
                sr = self.dst_node_list[torch.randint(len(self.dst_node_list), (num_samples, ),generator=self.rdm)]
                sl = self.local_dst[torch.randint(len(self.local_dst), (num_samples, ),generator=self.rdm)]
                s=torch.where(p<=self.prob,sr,sl)
                return s
            else:
                s = torch.randint(len(self.dst_node_list), (num_samples, ),generator=self.rdm)
                return self.dst_node_list[s]
        
class NeighborSampler():
    def __init__(
        self,
        num_nodes: int,
        num_layers: int,
        fanout: list,
        graph_data,
        workers = 1,
        tnb = None,
        is_distinct = 0,
        policy = "uniform",
        edge_weight: Optional[torch.Tensor] = None,
        graph_name = None,
        local_part = -1,
        node_part = None,
        edge_part = None,
        probability = 1,
        no_neg = False,
    ) -> None:
        r"""__init__
        Args:
            num_nodes: the num of all nodes in the graph
            num_layers: the num of layers to be sampled
            fanout: the list of max neighbors' number chosen for each layer
            workers: the number of threads, default value is 1
            tnb: neighbor infomation table
            is_distinct: 1-need distinct muti-edge, 0-don't need distinct muti-edge
            policy: "uniform" or "recent" or "weighted"
            edge_weight: the initial weights of edges
            graph_name: the name of graph
        should provide edge_index or (neighbors, deg)
        """
        super().__init__()
        self.num_layers = num_layers
        # 线程数不超过torch默认的omp线程数
        self.workers = workers # min(workers, torch.get_num_threads())
        self.fanout = fanout
        self.num_nodes = num_nodes
        self.graph_data=graph_data
        self.policy = policy
        self.is_distinct = is_distinct
        assert graph_name is not None
        self.graph_name = graph_name
        self.no_neg = no_neg
        if(tnb is None):
            if(graph_data.edge_ts is not None):
                timestamp,ind = graph_data.edge_ts.sort()
                timestamp = timestamp.contiguous()
                eid = graph_data.eid[ind].contiguous()
                row, col = graph_data.edge_index[:,ind]
            else:
                eid = graph_data.eid
                timestamp = None
                row, col = graph_data.edge_index
            if(edge_weight is not None):
                edge_weight = edge_weight.float().contiguous()
            self.tnb = starrygl.sampler_ops.get_neighbors(graph_name, row.contiguous(), col.contiguous(), num_nodes, is_distinct, eid, None,edge_weight, timestamp)
        else:
            assert tnb is not None
            self.tnb = tnb

        self.p_sampler = starrygl.sampler_ops.ParallelSampler(self.tnb, num_nodes, graph_data.num_edges, workers, 
                                         fanout, num_layers, policy, local_part,edge_part.to(torch.int),node_part.to(torch.int),probability)
        
    def _get_sample_info(self):
        return self.num_nodes,self.num_layers,self.fanout,self.workers
    
    def _get_sample_options(self):
        return {"is_distinct" : self.is_distinct,
                "policy" : self.policy,
                "with_eid" : self.tnb.with_eid, 
                "weighted" : self.tnb.weighted, 
                "with_timestamp" : self.tnb.with_timestamp}
    
    def insert_edges_with_timestamp(
            self, 
            edge_index : torch.Tensor, 
            eid : torch.Tensor, 
            timestamp : torch.Tensor,
            edge_weight : Optional[torch.Tensor] = None):
        row, col = edge_index
        # 更新节点数和tnb
        self.num_nodes = self.tnb.update_neighbors_with_time(
            row.contiguous(), 
            col.contiguous(), 
            timestamp.contiguous(), 
            eid.contiguous(), 
            self.is_distinct, 
            edge_weight.contiguous())
    
    def update_edges_weight(
            self, 
            edge_index : torch.Tensor, 
            eid : torch.Tensor,
            edge_weight : Optional[torch.Tensor] = None):
        row, col = edge_index
        # 更新tnb的权重信息
        if self.tnb.with_eid:
            self.tnb.update_edge_weight(
                eid.contiguous(),
                col.contiguous(),
                edge_weight.contiguous()
            )
        else:
            self.tnb.update_edge_weight(
                row.contiguous(),
                col.contiguous(),
                edge_weight.contiguous()
            )
    
    def update_nodes_weight(
            self, 
            nid : torch.Tensor,
            node_weight : Optional[torch.Tensor] = None):
        # 更新tnb的权重信息
        self.tnb.update_node_weight(
            nid.contiguous(),
            node_weight.contiguous()
        )

    def update_all_node_weight(
            self,
            node_weight : torch.Tensor):
        # 更新tnb的权重信息
        self.tnb.update_all_node_weight(node_weight.contiguous())

    def sample_from_nodes(
        self,
        nodes: torch.Tensor,
        ts: Optional[torch.Tensor] = None,
        is_unique = True,
        nid_mapper = None,
        eid_mapper = None,
        out_device = None
    ) -> SampleOutput:
        r"""Performs mutilayer sampling from the nodes specified in: nodes
        The specific number of layers is determined by parameter: num_layers
        returning a sampled subgraph in the specified output format: Tuple[torch.Tensor, list].

        Args:
            nodes: the list of seed nodes index,
            ts: the timestamp of nodes, optional,
            with_outer_sample: 0-sample in whole graph structure; 1-sample onehop outer nodel; 2-cross partition sampling
            fanout_index: optional. Specify the index to fanout
        Returns:
            sampled_nodes: the node sampled
            sampled_edge_index_list: the edge sampled
        """
        if self.policy != 'identity':
            self.p_sampler.neighbor_sample_from_nodes(nodes.contiguous(), ts.contiguous(), None)
            ret = self.p_sampler.get_ret()
        else:
            ret = None
        metadata = {}
        #print(nodes.shape[0],ret[0].src_index().max(),ret[0].src_index().min())
        return ret,metadata
        
    def sample_from_edges(
        self,
        edges: torch.Tensor,
        ets: Optional[torch.Tensor] = None,
        neg_sampling: Optional[NegativeSampling] = None,
        with_outer_sample: SampleType = SampleType.Whole,
        is_unique:bool = False,
        nid_mapper = None,
        eid_mapper = None,
        out_device = None
    ) -> SampleOutput:
        r"""Performs sampling from the edges specified in :obj:`index`,
        returning a sampled subgraph in the specified output format.

        Args:
            edges: the list of seed edges index
            with_outer_sample: 0-sample in whole graph structure; 1-sample onehop outer nodel; 2-cross partition sampling
            ets: the timestamp of edges, optional
            neg_sampling: The negative sampling configuration
        Returns:
            sampled_edge_index_list: the edges sampled
            sampled_eid_list: the edges' id sampled
            sampled_delta_ts_list:the edges' delta time sampled
            metadata: other infomation
        """
        src, dst = edges
        num_pos = src.numel()
        num_neg = 0
        with_timestap = ets is not None
        seed_ts = None

        if neg_sampling is not None:
            num_neg = math.ceil(num_pos * neg_sampling.amount)
            if neg_sampling.is_binary():
                src_neg,dst_neg = neg_sampling.sample(num_neg,self.num_nodes)
                #src_neg = neg_sampling.sample(num_neg, self.num_nodes)
                #dst_neg = neg_sampling.sample(num_neg, self.num_nodes)
                seed = torch.cat([src, dst, src_neg, dst_neg], dim=0)
                if with_timestap: # ts操作
                    seed_ts = torch.cat([ets, ets, ets.repeat(neg_sampling.amount),
                                         ets.repeat(neg_sampling.amount)], dim=0)
                    #seed_ts = torch.cat([ets, ets, ets, ets], dim=0)
            elif neg_sampling.is_triplet():
                dst_neg = neg_sampling.sample(num_neg, self.num_nodes)
                seed = torch.cat([src, dst, dst_neg], dim=0)
                if with_timestap: # ts操作
                    seed_ts = torch.cat([ets, ets, ets.repeat(neg_sampling.amount)], dim=0)

            elif neg_sampling.is_dygbinary():
                src_neg,dst_neg = neg_sampling.sample(num_neg,src,dst,
                                                      current_batch_start_time = ets.min(),
                                                      current_batch_end_time = ets.max())
                seed = torch.cat([src, dst, src_neg, dst_neg], dim=0)
                if with_timestap: # ts操作
                    seed_ts = torch.cat([ets, ets, ets.repeat(neg_sampling.amount),
                                         ets.repeat(neg_sampling.amount)], dim=0)
            
            elif neg_sampling.is_tgbtriplet():
                dst_neg = neg_sampling.sample(src,dst,ets)
                seed = torch.cat([src, dst, dst_neg], dim=0)
                if with_timestap: # ts操作
                    seed_ts = torch.cat([ets, ets, ets.repeat(neg_sampling.amount)], dim=0)

        else:
            seed = torch.cat([src, dst], dim=0)            
            if with_timestap: # ts操作
                seed_ts = torch.cat([ets, ets], dim=0)

        # 去重负采样
        """
        if neg_sampling is not None and neg_sampling.unique:
            if with_timestap: # ts操作
                pair, inverse_seed= torch.unique(torch.stack([seed, seed_ts],0), return_inverse=True, dim=1)
                seed, seed_ts = pair
                seed = seed.long()
            else:
                seed, inverse_seed = seed.unique(return_inverse=True)
        """
        if self.no_neg:
            out,metadata = self.sample_from_nodes(seed[:seed.shape[0]//3*2], seed_ts[:seed.shape[0]//3*2], is_unique=False)

        else:
            out,metadata = self.sample_from_nodes(seed, seed_ts, is_unique=False)
        src_pos_index = torch.arange(0,num_pos,dtype= torch.long,device=out_device)
        dst_pos_index = torch.arange(num_pos,2*num_pos,dtype= torch.long,device=out_device)
        if neg_sampling.is_triplet() or neg_sampling.is_tgbtriplet():
            dst_neg_index=torch.arange(2*num_pos,seed.shape[0],dtype= torch.long,device=out_device)
            src_neg_index = torch.tensor([],dtype= torch.long,device=out_device)
        else:
            src_neg_index = torch.arange(2*num_pos,3*num_pos,dtype= torch.long,device=out_device)
            dst_neg_index=torch.arange(3*num_pos,seed.shape[0],dtype= torch.long,device=out_device)
        metadata['seed'] = seed
        metadata['seed_ts'] = seed_ts
        metadata['src_pos_index']=src_pos_index
        metadata['dst_pos_index']=dst_pos_index
        if neg_sampling is not None :
            metadata['dst_neg_index'] = dst_neg_index
            if neg_sampling.is_triplet() or neg_sampling.is_tgbtriplet():
                pass
            else:
                metadata['src_neg_index'] = src_neg_index
        return out, metadata
    
    def sample_from_edges_with_distributed_dst(      
        self,data,id_mapper,
        neg_sampling: Optional[NegativeSampling] = None,
        with_outer_sample: SampleType = SampleType.Whole,
    ) -> SampleOutput:
        r"""Performs sampling from the edges specified in :obj:`index`,
        returning a sampled subgraph in the specified output format.

        Args:
            edges: the list of seed edges index
            with_outer_sample: 0-sample in whole graph structure; 1-sample onehop outer nodel; 2-cross partition sampling
            ets: the timestamp of edges, optional
            neg_sampling: The negative sampling configuration
        Returns:
            sampled_edge_index_list: the edges sampled
            sampled_eid_list: the edges' id sampled
            sampled_delta_ts_list:the edges' delta time sampled
            metadata: other infomation
        """

        num_pos = data.len
        num_neg = 0
        with_timestap = hasattr(data,'ts') and data.ts is not None
        seed_ts = None
        src,dst = data.edges
        #print('src is {} dst is {}'.format(src,dst))
        ets:torch.Tensor = data.ts
        if neg_sampling is not None:
            num_neg = math.ceil(num_pos * neg_sampling.amount)
            if neg_sampling.is_triplet():
                dst_neg = neg_sampling.sample(num_neg, self.num_nodes).to('cuda')
            elif neg_sampling.is_tgbtriplet:
                dst_neg = neg_sampling.sample(src,dst,ets).to('cuda')
            if with_timestap: # ts操作
                dst_ts = ets.repeat(neg_sampling.amount)
            #dst_data,dst_send_dict = DataSet.scatter_train_data(data,id_mapper)
            #print(dst_data.edges,dst_send_dict)
            train_dst_data  = DataSet(edges = torch.stack((src.tile(neg_sampling.amount+1),torch.cat((dst,dst_neg)))),
                                      eids = torch.nn.functional.pad(data.eids,(0,src.shape[0]*neg_sampling.amount)),
                                      ts = torch.cat((ets,dst_ts)),
                                      negs = torch.cat((torch.zeros(dst.shape[0],dtype =bool,device = dst.device),torch.ones(dst_neg.shape[0],dtype=bool,device=dst.device))))
            train_dst_data,dst_send_dict  = DataSet.scatter_train_data(train_dst_data,id_mapper)
            dst_data = train_dst_data[~train_dst_data.negs]
            neg_dst_data = train_dst_data[train_dst_data.negs]
            #neg_data = DataSet(edges = torch.stack((src.tile(neg_sampling.amount),dst_neg)),ts = dst_ts) 
            #neg_dst_data,_ = DataSet.scatter_train_data(neg_data,id_mapper)
        else:
            dst_data,dst_send_dict = DataSet.scatter_train_data(data,id_mapper)
        seed = torch.cat((src,dst_data.edges[1,:],neg_dst_data.edges[1,:]),dim = 0)
        seed_ts = torch.cat((ets,dst_data.ts,neg_dst_data.ts),dim = 0)
        # 去重负采样
        if neg_sampling is not None and neg_sampling.unique:
            if with_timestap: # ts操作
                pair, inverse_seed= torch.unique(torch.stack([seed, seed_ts],0), return_inverse=True, dim=1)
                seed, seed_ts = pair
                seed = seed.long()
            else:
                seed, inverse_seed = seed.unique(return_inverse=True)
        out = self.sample_from_nodes(seed.to('cpu'), seed_ts.to('cpu'), with_outer_sample)
        num_src = src.shape[0]
        num_dst = dst_data.len
        num_neg = neg_dst_data.len
        if neg_sampling is None or (not neg_sampling.unique):
            if with_timestap:
                return out, {'seed':seed,'seed_ts':seed_ts,
                             'src_pos_index':torch.arange(0,num_src), 
                             'dst_pos_index':torch.arange(num_src,num_src+num_dst), 
                             'dst_neg_index':torch.arange(num_dst+num_src,seed.shape[0]),
                             'dist_src_index':dst_data.edges[0,:],
                             'dist_neg_src_index':neg_dst_data.edges[0,:],
                             'dst_send_dict': dst_send_dict
                             },dst_data
            else:
                return out, {'seed':seed,
                             'src_pos_index':torch.arange(0,num_src), 
                             'dst_pos_index':torch.arange(num_src,num_src+num_dst), 
                             'dst_neg_index':torch.arange(num_dst+num_src,seed.shape[0]),
                             'dist_src_index':dst_data.edges[0,:],
                             'dist_neg_src_index':neg_dst_data.edges[0,:],
                             'dst_send_dict' : dst_send_dict},dst_data

        metadata = {}
        if neg_sampling.is_triplet() or neg_sampling.is_tgbtriplet():
            src_pos_index = inverse_seed[:num_src]
            dst_pos_index = inverse_seed[num_src: num_src + num_dst]
            dst_neg_index = inverse_seed[num_src+num_dst:]
            # src_index是seed里src点的索引
            # dst_pos_index是seed里dst_pos点的索引
            # dst_neg_index是seed里dst_neg点的索引
            metadata = {'seed':seed, 'src_pos_index':src_pos_index, 'dst_neg_index':dst_neg_index, 
                        'dst_pos_index':dst_pos_index,
                        'dist_src_index':dst_data.edges[0,:],
                        'dist_neg_src_index':neg_dst_data.edges[0,:],
                        'dst_send_dict' : dst_send_dict},dst_data
            if with_timestap:
                metadata['seed_ts'] = seed_ts
        # sampled_nodes最前方是原始序列的采样起点也就是去重后的seed
        return out, metadata,dst_data

if __name__=="__main__":
    # edge_index1 = torch.tensor([[0, 1, 1, 1, 2, 2, 2, 4, 4, 4, 5], # , 3, 3
    #                             [1, 0, 2, 4, 1, 3, 0, 3, 5, 0, 2]])# , 2, 5
    edge_index1 = torch.tensor([[0, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 5], # , 3, 3
                                [1, 0, 2, 0, 4, 1, 3, 0, 3, 3, 5, 0, 2]])# , 2, 5
    edge_weight1 = None
    timeStamp=torch.FloatTensor([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4])
    num_nodes1 = 6
    num_neighbors = 2
    # Run the neighbor sampling
    from Utils import GraphData
    g_data = GraphData(id=0, edge_index=edge_index1, timestamp=timeStamp, data=None, partptr=torch.tensor([0, num_nodes1]))
    sampler = NeighborSampler(num_nodes=num_nodes1, 
                              num_layers=3, 
                              fanout=[2, 1, 1], 
                              edge_weight=edge_weight1, 
                              graph_data=g_data, 
                              graph_name='a',
                              workers=4,
                              is_distinct = 0)

    out = sampler.sample_from_nodes(torch.tensor([1,2]),
                                    ts=torch.tensor([1, 2]),
                                    with_outer_sample=SampleType.Whole)
    # out = sampler.sample_from_edges(torch.tensor([[1,2],[4,0]]), 
    #                                 with_outer_sample=SampleType.Whole, 
    #                                 ets = torch.tensor([1, 2]))
    
    # Print the result
    print('node:', out.node)
    print('edge_index_list:', out.edge_index_list)
    print('eid_list:', out.eid_list)
    print('delta_ts_list:', out.delta_ts_list)
    print('metadata: ', out.metadata)
