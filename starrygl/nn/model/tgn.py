import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Tuple, List, Dict, Any
import math
from dgl.heterograph import DGLBlock
from starrygl.cache.cache_route import CacheRoute, CacheRouteManager
from starrygl.nn.hist.history import CompensationLayer
from starrygl.nn.model import EdgePredictor
from starrygl.nn.model.time_encoder import TimeEncoder
from ..gnn.graphattention import GraphAttention
from ..rnn.gru import GRUCell
from ..gnn.graphconv import GCNConv # 用于辅助或替换 GAT 中的 Conv

class TGN(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 time_feats: int,
                 edge_feats: int,
                 memory_params: Dict[str, Any],
                 dropout: float = 0.1,
                 att_dropout: float = 0.1,
                 num_heads: int = 2,
                 num_layers: int = 1,
                 cache_manager: CacheRouteManager = None
                ):
        """
        Temporal Graph Network (TGN) implementation.
        """
        super().__init__()
        self.edge_feats = edge_feats
        if memory_params is not None:
            self.time_enc = TimeEncoder(time_feats) if time_feats > 0 else None
            if memory_params['type'] == 'GRU':
                self.memory_updater = GRUCell(memory_params['input_dim'], hidden_size)
            elif memory_params['type'] == 'RNN':
                self.memory_updater = nn.RNNCell(memory_params['input_dim'], hidden_size)
            self.node_feat_map = torch.nn.Linear(input_size, hidden_size)
        else:
            self.memory_update = None
        self.attention = GraphAttention(
            in_features=input_size if self.memory_updater is None else hidden_size,
            out_features=hidden_size,
            dim_time=time_feats,
            dim_edge_feat= edge_feats,
            num_head=num_heads,  # 通常 TGN 使用单头注意力
            dropout=dropout,  # TGN 中通常不使用 dropout
            att_dropout=att_dropout,
            num_layers= num_layers,
        )
        self.comp_time_enc = TimeEncoder(hidden_size) # 时间编码维度设为 hidden_size 以便融合
        self.mem_compensation = CompensationLayer(
            feature_dim=hidden_size, 
            time_dim=hidden_size,
            hidden_dim=hidden_size
        )
        
        # GNN 隐层的补偿 (Layer > 0 输入)
        self.gnn_compensation = CompensationLayer(
            feature_dim=hidden_size,
            time_dim=hidden_size,
            hidden_dim=hidden_size
        )
        
    def memory_upd(self, dist_flag:torch.Tensor, 
                         memory:torch.Tensor, 
                         mem_ts:torch.Tensor, 
                         mailbox:torch.Tensor, 
                         mail_ts:torch.Tensor):
        
        time_feat = self.time_enc(mail_ts - mem_ts) if self.time_enc is not None else None
        if time_feat is not None:
            mem_input = torch.cat([mailbox, time_feat], dim=1) 
        else:
            mem_input = mailbox
        out_memory = self.memory_updater(mem_input, memory)
        out_ts = mail_ts
        return out_memory, out_ts

    
    def _apply_compensation(self, 
                            comp_layer: CompensationLayer,
                            h_fresh: torch.Tensor,
                            h_hist_data: Any,
                            is_remote: torch.Tensor, 
                            current_ts: torch.Tensor) -> torch.Tensor:
        """
        执行补偿逻辑的核心函数。
        """
        if h_hist_data is None:
            return h_fresh

        if isinstance(h_hist_data, (list, tuple)):
            h_hist, hist_ts = h_hist_data
        else:
            h_hist = h_hist_data   
            hist_ts = None

        if h_hist.shape[-1] != h_fresh.shape[-1]:
            return h_fresh

        mask = is_remote.unsqueeze(-1) if is_remote.dim() == 1 else is_remote
        if not mask.any():
            return h_fresh
        if hist_ts is not None:
            dt = (current_ts - hist_ts).clamp(min=0)
        else:
            dt = torch.zeros_like(current_ts, dtype=torch.float32)
            
        h_comp = comp_layer(h_hist, dt, self.comp_time_enc)
        return torch.where(mask, h_comp, h_fresh)
    
    def forward(self, 
                blocks: List[DGLBlock], 
                x: torch.Tensor,
                memory_data: Tuple[torch.Tensor], # (mem, mem_ts)
                mailbox_data: Tuple[torch.Tensor],# (mailbox, mail_ts) - 从外部传入，不用再生成
                history_data: List[Any] = None,
                dist_flags: List[torch.Tensor] = None, 
                nid_mapper: List[torch.Tensor] = None, #可能不要了？
                upd_hook: Callable = None,
                ) -> torch.Tensor:
        
        # === 1. Memory Handling ===
        if self.memory_updater is not None:
            m_val, m_ts = memory_data if memory_data else (None, None)
            mail_val, mail_ts = mailbox_data if mailbox_data else (None, None)
            out_memory, out_ts = self.memory_upd(
                dist_flags[0], m_val, m_ts, mail_val, mail_ts
            )
            if upd_hook is not None and nid_mapper is not None:
                src_nids = nid_mapper[0] 
                upd_hook(0, src_nids, out_memory.detach(), out_ts.detach())
            if self.node_feat_map is not None:
                h_feat = self.node_feat_map(x)
                h = out_memory + h_feat # Add (ResNet style)
            else:
                h = out_memory
        else:
            out_memory = None
            out_ts = None
            h = g[0].srcdata['x']
            if self.node_feat_map is not None:
                h = self.node_feat_map(h)
        
        # === 2. GNN Layers ===
        for i, attention_layer in enumerate(self.attention.convs):
            block = g[i]
            h_prev = h
            
            h_hist = history_data[i] if history_data else None
            is_remote = dist_flags[i] if dist_flags else None
            curr_ts = block.srcdata.get('ts', m_ts if i == 0 else None)
            if i == 0:
                comp_layer = self.mem_compensation
            else:
                comp_layer = self.gnn_compensation
            h_input = self._apply_compensation(
                comp_layer=comp_layer,
                h_fresh=h_prev,
                h_hist_data=h_hist,
                is_remote=is_remote,
                current_ts=curr_ts
            )
            block.dstdata['x'] = h_input[:block.num_dst_nodes()]
            h = attention_layer.async_forward(block, h_input)       
            if upd_hook is not None and i < len(self.attention.convs) - 1:
                with torch.no_grad():
                    dst_indices = block.dstdata['nid']
                    global_dst_nids = nid_mapper[0][dst_indices]
                    dst_ts = block.dstdata['ts']
                    upd_hook(i + 1, global_dst_nids, h.detach(), dst_ts.detach())
            if i < len(self.attention.convs) - 1:
                h = F.relu(h)
        with torch.no_grad():
            self.last_memory = out_memory
        return h