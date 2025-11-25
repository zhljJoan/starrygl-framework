import torch
from torch import Tensor

class ChunkAwareTemporalSampler(HybridTemporalSampler):
    """支持chunk划分的时空采样器"""
    
    def __init__(self, ctcsr_index, chunk_manager, time_strategy: str = "chunk_aware"):
        super().__init__(ctcsr_index, time_strategy)
        self.chunk_manager = chunk_manager
        self.chunk_sampling_cache = {}
        
    def sample_with_chunk_constraint(self,
                                   nodes: Tensor,
                                   timestamps: Tensor,
                                   fanouts: List[int],
                                   chunk_constraint: str = "intra_first") -> Dict:
        """基于chunk约束的采样"""
        
        sampled_results = {
            'intra_chunk': [],
            'cross_chunk': [],
            'chunk_boundaries': []
        }
        
        for i, (node, timestamp) in enumerate(zip(nodes, timestamps)):
            # 确定节点所属chunk
            chunk_id = self.chunk_manager.get_node_chunk(node)
            
            # 分阶段采样：先chunk内，再跨chunk
            if chunk_constraint == "intra_first":
                intra_samples = self.sample_intra_chunk(node, chunk_id, timestamp, fanouts[0])
                cross_samples = self.sample_cross_chunk(node, chunk_id, timestamp, fanouts[1])
            else:
                # 其他采样策略
                intra_samples, cross_samples = self.unified_chunk_sampling(
                    node, chunk_id, timestamp, fanouts
                )
            
            sampled_results['intra_chunk'].append(intra_samples)
            sampled_results['cross_chunk'].append(cross_samples)
            sampled_results['chunk_boundaries'].append(chunk_id)
        
        return sampled_results
    
    def sample_intra_chunk(self, 
                          node: int, 
                          chunk_id: int, 
                          timestamp: float,
                          fanout: int) -> Dict:
        """chunk内采样 - 利用局部性"""
        
        # 检查缓存
        cache_key = (chunk_id, timestamp)
        if cache_key in self.chunk_sampling_cache:
            chunk_neighbors = self.chunk_sampling_cache[cache_key]
        else:
            # 查询chunk内所有节点的时序邻居
            chunk_nodes = self.chunk_manager.get_chunk_nodes(chunk_id)
            chunk_neighbors = self.ctcsr_index.batch_query_temporal_neighbors(
                chunk_nodes, timestamp, time_delta=self.time_delta
            )
            self.chunk_sampling_cache[cache_key] = chunk_neighbors
        
        # 从chunk邻居中采样特定节点
        if node in chunk_neighbors:
            neighbors = chunk_neighbors[node]
            sampled = self._sample_from_neighbors(neighbors, fanout)
            return {
                'nodes': sampled['nodes'],
                'timestamps': sampled['timestamps'],
                'chunk_ids': torch.full_like(sampled['nodes'], chunk_id)
            }
        else:
            return {'nodes': torch.tensor([]), 'timestamps': torch.tensor([]), 'chunk_ids': torch.tensor([])}
    
    def sample_cross_chunk(self,
                          node: int,
                          chunk_id: int,
                          timestamp: float,
                          fanout: int) -> Dict:
        """跨chunk采样 - 处理远程依赖"""
        
        # 获取与该chunk有连接的远程chunk
        connected_chunks = self.chunk_manager.get_connected_chunks(chunk_id)
        
        cross_samples = []
        for remote_chunk in connected_chunks:
            # 查询跨chunk边
            cross_edges = self.ctcsr_index.query_cross_chunk_edges(
                node, chunk_id, remote_chunk, timestamp
            )
            
            if len(cross_edges) > 0:
                sampled = self._sample_from_neighbors(cross_edges, fanout // len(connected_chunks))
                cross_samples.append(sampled)
        
        if cross_samples:
            # 合并跨chunk采样结果
            combined = self._combine_cross_chunk_samples(cross_samples)
            combined['chunk_ids'] = torch.tensor([c for sample in cross_samples 
                                                for c in sample.get('chunk_ids', [])])
            return combined
        else:
            return {'nodes': torch.tensor([]), 'timestamps': torch.tensor([]), 'chunk_ids': torch.tensor([])}