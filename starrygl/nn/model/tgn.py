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
        
    def memory_upd(self, 
                         memory:torch.Tensor, 
                         mem_ts:torch.Tensor, 
                         mailbox:torch.Tensor, 
                         mail_ts:torch.Tensor):
        
        time_feat = self.time_enc(mail_ts.reshape(-1) - mem_ts) if self.time_enc is not None else None
        if time_feat is not None:
            mem_input = torch.cat([mailbox.reshape(mailbox.shape[0], -1), time_feat], dim=1) 
        else:
            mem_input = mailbox.reshape(mailbox.shape[0], -1)
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
        if h_fresh.shape[0] != is_remote.shape[0]:
            print(f"维度不匹配! h_fresh: {h_fresh.shape[0]}, is_remote: {is_remote.shape[0]}")
        if h_hist_data is None:
            return h_fresh

        if isinstance(h_hist_data, tuple):
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
                routes: List[CacheRoute],
                mailbox_data: Tuple[torch.Tensor, torch.Tensor] = None,
                upd_hook: Callable = None,
                ) -> torch.Tensor:
        
        # === 1. Memory Handling ===
        if self.memory_updater is not None:
            with torch.no_grad():
                x = blocks[0].x_future.wait()
                m_val, m_ts = blocks[0].h_future_mem.wait()
                mail_val, mail_ts = mailbox_data.wait() if mailbox_data else (None, None)
            out_memory, out_ts = self.memory_upd(
                 m_val, m_ts, mail_val, mail_ts
            )
            with torch.no_grad():
                self.last_memory = out_memory
                self.last_ts = m_ts
                h_mem_hist = blocks[0].h_future_mem.wait()
                is_mem_remote = blocks[0].is_remote_mem
                if upd_hook is not None:
                    #新计算出来的memory更新 & 同步，放在CPU还是GPU端，会不会打断训练，异步更新的接口，需要广播？
                    upd_hook(0, out_memory.detach(), out_ts, routes[0], blocks[0].srcdata[dgl.NID])
            if self.node_feat_map is not None:
                h_feat = self.node_feat_map(x)
                h = out_memory + h_feat # Add (ResNet style)
            else:
                h = out_memory
        else:
            with torch.no_grad():
                out_memory, out_ts = None, None
                x = blocks[0].x_future.wait()
                h = x
            if self.node_feat_map is not None:
                h = self.node_feat_map(h)
        
        # === 2. GNN Layers ===
        for i, attention_layer in enumerate(self.attention.convs):
            with torch.no_grad():
                block = blocks[i]
                h_prev = h
                e_future = block.e_future.wait()
                block.edata['f'] = e_future
                curr_ts = block.srcdata.get('ts', m_ts if i == 0 else None)
            if i == 0 and self.memory_updater is not None:
                comp_layer = self.mem_compensation
                h_input = self._apply_compensation(
                    comp_layer=comp_layer,
                    h_fresh=h_prev,
                    h_hist_data=h_mem_hist,
                    is_remote=is_mem_remote,
                    current_ts=curr_ts
                )
            block.dstdata['x'] = h_input[block.src_indices]
            h = attention_layer(block, h_input)      
            #print('block dstdata {} h shape {} block ts {}\n'.format(block.dstdata['x'].shape, h.shape, block.dstdata['ts'].shape)) 
            with torch.no_grad():
                if upd_hook is not None and i < len(self.attention.convs) - 1:
                    r_i = i + 1
                    upd_hook(r_i, h.detach(), block.dstdata['ts'], routes[r_i], block.dstdata[dgl.NID])
                comp_layer = self.gnn_compensation
            
                next_block = blocks[i + 1] if i + 1 < len(blocks) else None
                h_hist_next = block.h_future.wait()
                is_remote_next = block.is_remote
                curr_ts = block.dstdata['ts']
            #print('apply compensation at layer {} {} {} {} {} {}'.format(i, block.src_indices.shape, h.shape, h_hist_next[0].shape, h_hist_next[1].shape, is_remote_next.shape, curr_ts.shape))
            h = self._apply_compensation(
                comp_layer=comp_layer,
                h_fresh=h,
                h_hist_data=h_hist_next,
                is_remote=is_remote_next,
                current_ts=curr_ts
            )
            if i < len(self.attention.convs) - 1:
                h = F.relu(h)

        return h