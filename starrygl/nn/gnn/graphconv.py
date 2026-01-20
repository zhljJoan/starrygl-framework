import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

from torch import Tensor
from typing import *

from dgl import DGLGraph

from starrygl.route.route import Route
from starrygl.nn.async_module import AsyncModule



class GCNConv(AsyncModule):
    def __init__(self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        shortcut: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.shortcut = bool(shortcut)
        
        w_in = self.in_features * 2 if shortcut else self.in_features
        w_out = self.out_features

        self.weight = nn.Parameter(torch.empty(w_in, w_out))
        if bias:
            self.bias = nn.Parameter(torch.empty(w_out))
        else:
            self.bias = None
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @torch.no_grad()
    @staticmethod
    def gcn_norm(g: DGLGraph) -> Tensor:
        if 'gcn_norm' in g.edata:
            return g.edata['gcn_norm']
        
        if g.is_block:
            if 'w' in g.edata:
                with g.local_scope():
                    g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 's'))
                    g.apply_edges(fn.e_div_v('w', 's', 'x'))
                    x = g.edata['x']
            else:
                with g.local_scope():
                    g.dstdata['s'] = g.in_degrees()
                    g.apply_edges(fn.e_div_v('w', 's', 'x'))
                    x = g.edata['x']
        else:
            if 'w' in g.edata:
                r = dgl.reverse(g, copy_ndata=False, copy_edata=True)
                r.update_all(fn.copy_e('w', 'm'), fn.sum('m', 's'))
                r = r.ndata['s']

                with g.local_scope():
                    g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 's'))
                    g.ndata['t'] = r
                    g.apply_edges(fn.u_mul_v('t', 's', 'x'))
                    x = g.edata['w'] * torch.rsqrt(g.edata['x'])
            else:
                with g.local_scope():
                    g.ndata['s'] = g.in_degrees().float()
                    g.ndata['t'] = g.out_degrees().float()
                    g.apply_edges(fn.u_mul_v('t', 's', 'x'))
                    x = torch.rsqrt(g.edata['x'])
        x = x.nan_to_num_(0.0)
        return x
    
    @staticmethod
    def msg_pass(g: DGLGraph, x: Tensor) -> Tensor:
        with g.local_scope():
            if g.is_block:
                g.srcdata['x'] = x
            else:
                g.ndata['x'] = x
            
            if 'gcn_norm' not in g.edata:
                g.edata['gcn_norm'] = GCNConv.gcn_norm(g)
            
            g.update_all(fn.u_mul_e('x', 'gcn_norm', 'm'), fn.sum('m', 'x'))
            
            if g.is_block:
                x = g.dstdata['x']
            else:
                x = g.ndata['x']
        return x
    
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
    
    async def async_forward(self,
        g: DGLGraph,
        x: Tensor,
        route: Route | None = None,
    ) -> Tensor:
        _x = x

        if route is None:
            _ = await self.yield_forward() # yield control flow to other tasks
        else:
            x = await route.async_forward(x)

        x = self.msg_pass(g, x)
        
        if self.shortcut:
            x = torch.cat([x, _x[:x.size(0)]], dim=-1)
        
        x = x @ self.weight
        if isinstance(self.bias, Tensor):
            x = x + self.bias
        return x


class GCN(AsyncModule):
    def __init__(self,
        in_features: int,
        out_features: int,
        num_layers: int = 1,
        bias: bool = True,
        shortcut: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_layers = int(num_layers)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = self.in_features if i == 0 else self.out_features
            out_ch = self.out_features
            self.convs.append(GCNConv(in_ch,out_ch, bias=bias, shortcut=shortcut))
    
    def reset_parameters(self) -> None:
        for conv in self.convs:
            cast(GCNConv, conv).reset_parameters()

    async def async_forward(self, g: DGLGraph, x: Tensor | None = None) -> Tensor:
        x, route, route_first = GCNConv.get_inputs(g, x)

        for i, conv in enumerate(self.convs):
            conv = cast(GCNConv, conv)
            if i == 0:
                r = route if route_first else None
                x = await conv.async_forward(g, x, route=r)
            else:
                x = F.relu(x)
                r = route
                x = await conv.async_forward(g, x, route=r)
        return x
    


