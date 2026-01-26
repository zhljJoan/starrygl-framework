
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

from torch import Tensor
from typing import *

from dgl import DGLGraph

from starrygl.nn.model.time_encoder import TimeEncoder
from starrygl.route.route import Route
from starrygl.nn.async_module import AsyncModule

class TransfomerAttentionLayer(AsyncModule):

    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, combined=False):
        super(TransfomerAttentionLayer, self).__init__()
        self.num_head = num_head
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.dropout = torch.nn.Dropout(dropout)
        self.att_dropout = torch.nn.Dropout(att_dropout)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.combined = combined
        if dim_time > 0:
            self.time_enc = TimeEncoder(dim_time)
        if dim_node_feat + dim_time > 0:
            self.w_q = torch.nn.Linear(dim_node_feat + dim_time, dim_out)
        self.w_k = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
        self.w_v = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
        self.w_out = torch.nn.Linear(dim_node_feat + dim_out, dim_out)
        self.layer_norm = torch.nn.LayerNorm(dim_out)

    @staticmethod
    def get_inputs(
        g: DGLGraph,
        x: Tensor | None = None,
    ) -> Tuple[Tensor, Route | None, bool]:
        route: Route | None = getattr(g, "route", None)

        if x is None:
            if g.is_block:
                if 'x' in g.dstdata:
                    route_first = True
                    x = g.dstdata['x']
                else:
                    route_first = False
                    x = g.srcdata['x']
            else:
                x = g.ndata['x']
                route_first = False
        else:
            if g.is_block:
                if x.size(0) == g.num_src_nodes():
                    route_first = False
                else:
                    assert x.size(0) == g.num_dst_nodes()
                    route_first = True
            else:
                assert x.size(0) == g.num_nodes()
                route_first = False

        return x, route, route_first
    
    def forward(self,
                    g:DGLGraph, 
                    x: Tensor | None = None,
                 ) -> Tensor:
        
        with g.local_scope():
            if self.dim_time > 0:
                src_time_feat = self.time_enc(g.edata['dt'])
                dst_time_feat = self.time_enc(torch.zeros(g.num_dst_nodes(), dtype=torch.float32, device=g.device))
            else:
                src_time_feat = torch.empty((g.num_edges(), 0), self.dim_time, device=g.device)
                dst_time_feat = torch.empty((g.num_dst_nodes(), 0), self.dim_time, device=g.device)
            if 'f' not in g.edata and self.dim_edge_feat > 0:
                edge_feat = torch.empty(g.num_edges(), self.dim_edge_feat, device=g.device)
            
            inputs = torch.cat((x[g.adj_tensors('csc')[1]],g.edata['f'], src_time_feat), dim=1) if x is not None else torch.cat((edge_feat, src_time_feat), dim=1)
            query = torch.cat((g.dstdata['x'], dst_time_feat), dim=1) if x is not None else dst_time_feat
            Q = self.w_q(query) if self.dim_node_feat + self.dim_time > 0 else torch.ones((g.num_dst_nodes(), self.dim_out), device=g.device)
            K = self.w_k(inputs)
            V = self.w_v(inputs)
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            _, dst_idx = g.edges()
            score = torch.sum(Q[dst_idx] * K, dim=2) / (self.dim_out // self.num_head) ** 0.5
            att = dgl.ops.edge_softmax(g, self.att_act(score))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            g.edata['v'] = V
            g.update_all(dgl.function.copy_e('v', 'm'), dgl.function.sum('m', 'h'))
            if self.dim_node_feat != 0:
                rst = torch.cat([g.dstdata['h'], g.dstdata['x']], dim=1)
            else:
                rst = g.dstdata['h']
            rst = self.w_out(rst)
            rst = torch.nn.functional.relu(self.dropout(rst))
        return self.layer_norm(rst)
    
    async def async_forward(self,
                    g:DGLGraph, 
                    x: Tensor | None = None,
                    route: Route|None = None,
                 ) -> Tensor:
        if route is None:
            _ = await self.yield_forward()
        else:
            x = await route.async_forward(x)
        
        with g.local_scope():
            if self.dim_time > 0:
                src_time_feat = self.time_enc(g.edata['dt'])
                dst_time_feat = self.time_enc(torch.zeros(g.num_dst_nodes(), dtype=torch.float32, device=g.device))
            else:
                src_time_feat = torch.empty((g.num_edges(), 0), self.dim_time, device=g.device)
                dst_time_feat = torch.empty((g.num_dst_nodes(), 0), self.dim_time, device=g.device)
            if edge_feat is None and self.dim_edge_feat > 0:
                edge_feat = torch.empty(g.num_edges(), self.dim_edge_feat, device=g.device)
            
            inputs = torch.cat((x,g.edata['f'], src_time_feat), dim=1) if x is not None else torch.cat((edge_feat, src_time_feat), dim=1)
            query = torch.cat((g.dstdata['x'], dst_time_feat), dim=1) if x is not None else dst_time_feat
            Q = self.w_q(query) if self.dim_node_feat + self.dim_time > 0 else torch.ones((g.num_dst_nodes(), self.dim_out), device=g.device)
            K = self.w_k(inputs)
            V = self.w_v(inputs)
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            att = dgl.ops.edge_softmax(g, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            g.edata['v'] = V
            g.update_all(dgl.function.copy_e('v', 'm'), dgl.function.sum('m', 'h'))
            if self.dim_node_feat != 0:
                rst = torch.cat([g.dstdata['h'], g.dstdata['x']], dim=1)
            else:
                rst = g.dstdata['h']
            rst = self.w_out(rst)
            rst = torch.nn.functional.relu(self.dropout(rst))
        return self.layer_norm(rst)
        
    
class GraphAttention(AsyncModule):
    def __init__(self,
        in_features: int,
        out_features: int,
        dim_time: int,
        dim_edge_feat: int,
        num_head: int,
        dropout: float,
        att_dropout: float,
        num_layers: int = 1
    ) -> None:
        super().__init__()
        
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.dim_time = int(dim_time)
        self.dim_edge_feat = int(dim_edge_feat)
        self.num_layers = int(num_layers)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(TransfomerAttentionLayer(
                dim_node_feat = in_features if i == 0 else out_features,
                dim_edge_feat = dim_edge_feat,
                dim_time = dim_time,
                num_head = num_head,
                dropout = dropout,
                att_dropout = att_dropout,
                dim_out = out_features
                )
            )
        
    
    def reset_parameters(self) -> None:
        for conv in self.convs:
            cast(TransfomerAttentionLayer, conv).reset_parameters()

    async def async_forward(self, g: DGLGraph, x: Tensor | None = None) -> Tensor:
        x, route, route_first = TransfomerAttentionLayer.get_inputs(g, x)

        for i, conv in enumerate(self.convs):
            conv = cast(TransfomerAttentionLayer, conv)
            if i == 0:
                r = route if route_first else None
                x = await conv.async_forward(g, x, route=r)
            else:
                x = F.relu(x)
                r = route
                x = await conv.async_forward(g, x, route=r)
        return x
    
    def forward(self, g: DGLGraph, x: Tensor | None = None) -> Tensor:
        for i, conv in enumerate(self.convs):
            conv = cast(TransfomerAttentionLayer, conv)
            if i == 0:
                x = conv.forward(g, x)
            else:
                x = F.relu(x)
                x = conv.forward(g, x)
        return x
