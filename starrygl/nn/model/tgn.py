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
        
    def memory_upd(self, memory, mem_ts, mailbox, mail_ts):
        # 使用 NVTX 标记 Memory 更新的核心计算
        with torch.cuda.nvtx.range("TGN_Memory_Upd_Compute"):
            time_feat = self.time_enc(mail_ts.reshape(-1) - mem_ts) if self.time_enc is not None else None
            if time_feat is not None:
                mem_input = torch.cat([mailbox.reshape(mailbox.shape[0], -1), time_feat], dim=1) 
            else:
                mem_input = mailbox.reshape(mailbox.shape[0], -1)
            out_memory = self.memory_updater(mem_input, memory)
            return out_memory, mail_ts
    
    @torch.compile
    def _apply_compensation(self, comp_layer, h_fresh, h_hist_data, is_remote, current_ts):
        # 标记补偿逻辑的触发
        with torch.cuda.nvtx.range("TGN_Apply_Compensation"):
            if h_hist_data is None: return h_fresh
            
            # 解析历史数据
            h_hist, hist_ts = h_hist_data if isinstance(h_hist_data, tuple) else (h_hist_data, None)
            if h_hist.shape[-1] != h_fresh.shape[-1]: return h_fresh

            mask = is_remote.unsqueeze(-1) if is_remote.dim() == 1 else is_remote
            if not mask.any(): return h_fresh
            
            dt = (current_ts - hist_ts).clamp(min=0) if hist_ts is not None else torch.zeros_like(current_ts)
            
            # 补偿层前向计算
            h_comp = comp_layer(h_hist, dt.float(), self.comp_time_enc)
            return torch.where(mask, h_comp, h_fresh)

    def forward(self, blocks, routes, mailbox_data=None, upd_hook=None):
        
        # === 1. Memory Handling ===
        with torch.cuda.nvtx.range("TGN_Stage1_Memory"):
            if self.memory_updater is not None:
                # 显式标记数据等待时间（DMA/磁盘/预取）
                with torch.cuda.nvtx.range("Wait_Input_Futures"):
                    x = blocks[0].x_future.wait()
                    m_val, m_ts = blocks[0].h_future_mem.wait()
                    mail_val, mail_ts = mailbox_data.wait() if mailbox_data else (None, None)
                
                out_memory, out_ts = self.memory_upd(m_val, m_ts, mail_val, mail_ts)
                
                with torch.no_grad():
                    self.last_memory = out_memory
                    self.last_ts = m_ts
                    h_mem_hist = blocks[0].h_future_mem.wait()
                    is_mem_remote = blocks[0].is_remote_mem
                    
                    if upd_hook is not None:
                        with torch.cuda.nvtx.range("Memory_Update_Hook"):
                            upd_hook(0, out_memory.detach(), out_ts, routes[0], blocks[0].srcdata[dgl.NID])
                
                h_feat = self.node_feat_map(x) if self.node_feat_map else 0
                h = out_memory + h_feat
            else:
                x = blocks[0].x_future.wait()
                h = self.node_feat_map(x) if self.node_feat_map else x

        # === 2. GNN Layers ===
        for i, attention_layer in enumerate(self.attention.convs):
            # 为每一层 GNN 创建独立的 NVTX Range
            with torch.cuda.nvtx.range(f"TGN_GNN_Layer_{i}"):
                block = blocks[i]
                
                with torch.cuda.nvtx.range(f"Layer_{i}_Wait_Edges"):
                    e_future = block.e_future.wait()
                    block.edata['f'] = e_future
                    curr_ts = block.srcdata.get('ts', m_ts if i == 0 else None)

                # 第一层的 Memory 补偿
                if i == 0 and self.memory_updater is not None:
                    h = self._apply_compensation(self.mem_compensation, h, h_mem_hist, is_mem_remote, curr_ts)

                # GNN Conv 计算
                block.dstdata['x'] = h[block.src_indices]
                with torch.cuda.nvtx.range(f"Layer_{i}_Attention_Compute"):
                    h = attention_layer(block, h)      

                # 更新 Hook 与下一层补偿
                with torch.no_grad():
                    if upd_hook is not None and i < len(self.attention.convs) - 1:
                        with torch.cuda.nvtx.range(f"Layer_{i}_Update_Hook"):
                            upd_hook(i + 1, h.detach(), block.dstdata['ts'], routes[i+1], block.dstdata[dgl.NID])
                    
                    with torch.cuda.nvtx.range(f"Layer_{i}_Comp_Data_Wait"):
                        h_hist_next = block.h_future.wait()
                        is_remote_next = block.is_remote
                        curr_ts_next = block.dstdata['ts']
                
                # 节点特征补偿
                h = self._apply_compensation(self.gnn_compensation, h, h_hist_next, is_remote_next, curr_ts_next)
                
                if i < len(self.attention.convs) - 1:
                    h = F.relu(h)

        return h