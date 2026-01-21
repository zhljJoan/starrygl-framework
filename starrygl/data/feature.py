import torch
from torch import Tensor
from typing import List, Optional, Tuple, Union

from starrygl.utils.partition_book import DistRouteIndex, IDMapper, PartitionState



class DistFeatureCache:
    """
    分布式特征缓存封装
    支持基于 IDMapper 的自动 ID 转换和特征索引
    """
    def __init__(self, 
                 state: IDMapper, 
                 features: Tensor, 
                 name: str = "node_feat"):
        self.state = state
        self.name = name
        if not features.is_contiguous():
            features = features.contiguous()
            
        if features.device.type == 'cpu' and not features.is_pinned():
            self.features = features.pin_memory()
        else:
            self.features = features
            
        self.loc_nums = self.state.loc_nums

    @property
    def device(self):
        return self.features.device

    @property
    def feat_dim(self):
        return self.features.size(1)

    def __getitem__(self, key: Union[Tensor, DistRouteIndex, tuple]) -> Tensor:
        if isinstance(key, tuple):
            gids, sub_idx = key[0], key[1:]
        else:
            gids, sub_idx = key, None

        loc_indices = self.state.to_local(gids, device=self.features.device)
        feat = self.features[loc_indices]
        if sub_idx is not None:
            return feat[(slice(None), *sub_idx)]
        
        return feat

    def update_features(self, local_indices: Tensor, new_feats: Tensor):
        """更新本地缓存的特征"""
        self.features[local_indices] = new_feats.to(self.device)

    def get_master_features(self) -> Tensor:
        """获取仅属于当前分区的 Master 节点特征"""
        num_master = self.state.num_master
        return self.features[:num_master]

    def __repr__(self):
        return (f"DistFeatureCache(name={self.name}, "
                f"storage={self.features.shape}, device={self.device})")
        
    
    def prefetch(self, gids: Tensor, stream: Optional[torch.cuda.Stream] = None):
        if self.features.device.type == 'cpu':
            with torch.cuda.stream(stream):
                loc_idx = self.state.to_local(gids, device=torch.device('cpu'))
                return self.features[loc_idx].cuda(non_blocking=True)
        return self[gids]