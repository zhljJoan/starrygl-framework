from mailbox import Mailbox
from typing import Optional, Union
from torch import Tensor
from starrygl.data.feature import DistFeatureCache
from starrygl.utils.partition_book import DistRouteIndex, PartitionState


class StarryglGraphContext:
    def __init__(self, 
                 state: PartitionState, 
                 node_feats: Optional[Tensor] = None,
                 edge_feats: Optional[Tensor] = None,
                 mailbox_size: int = None, 
                 dim_out: int = None, 
                 dim_edge_feat: int = None
                 ):
        
        self.state = state
        self.device = state.node_mapper.device
        
        # 1. 自动封装特征访问
        self.node_feats = DistFeatureCache(state.node_mapper, node_feats, "node") if node_feats is not None else None
        self.edge_feats = DistFeatureCache(state.edge_mapper, edge_feats, "edge") if edge_feats is not None else None
        
        # 2. 自动封装 Mailbox
        self.mailbox = None
        if dim_out is not None and dim_out > 0:
            self.mailbox = Mailbox(state, mailbox_size, dim_out, dim_edge_feat, device=self.device)

    def to(self, device):
        self.state.node_mapper.loc_ids = self.state.node_mapper.loc_ids.to(device)
        if self.node_feats: self.node_feats.features = self.node_feats.features.to(device)
        if self.mailbox: self.mailbox.to(device)
        self.device = device
        return self
