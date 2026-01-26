from typing import Tuple, Union, Optional
import torch
from torch import Tensor
import torch_scatter

from starrygl.utils.context import DistributedContext
from starrygl.utils.partition_book import DistRouteIndex, PartitionState

class Mailbox:
    def __init__(self, num_nodes, mailbox_size, dim_out, dim_edge_feat, 
                 _next_mail_pos=None, _update_mail_pos=False, device=None):
        self.num_nodes = num_nodes
        # 记录逻辑设备，如果是 None 默认先在 CPU 初始化
        self.device = device if device is not None else torch.device('cpu')
        self.mailbox_size = mailbox_size
        self.dim_out = dim_out
        self.dim_edge_feat = dim_edge_feat
        
        feat_dim = 2 * dim_out + dim_edge_feat
        
        # --- 高性能优化：开启 pin_memory ---
        # 只有在 CPU 上的 Tensor 才能被 Pin
        is_cpu = self.device.type == 'cpu'
        
        self.mailbox = torch.zeros(
            self.num_nodes, mailbox_size, feat_dim,
            device=self.device, dtype=torch.float32,
            pin_memory=is_cpu  # 关键优化：启用锁页内存
        )
        self.mailbox_ts = torch.zeros(
            (self.num_nodes, mailbox_size), 
            device=self.device, dtype=torch.long,
            pin_memory=is_cpu
        )
        
        self.update_stream = torch.cuda.Stream() if self.device.type == 'cuda' else None
        # 索引通常较小，根据需要选择是否 pin
        self.next_mail_pos = torch.zeros((num_nodes), dtype=torch.long, device=self.device) \
                             if _next_mail_pos is None else _next_mail_pos.to(self.device)
        
        self.update_mail_pos = _update_mail_pos

    def reset(self):
        self.mailbox.fill_(0)
        self.mailbox_ts.fill_(0)
        self.next_mail_pos.fill_(0)
        
    @property
    def shape(self):
        return (self.num_nodes, 
                self.mailbox_size, 
                2 * self.dim_out + self.dim_edge_feat)
    

    
    def move_to_gpu(self):
        """利用 pin_memory 带来的 non_blocking 特性快速搬运"""
        ctx_device = DistributedContext.get_default_context().get_device()
        # 如果之前是 pinned memory，这里的 non_blocking 会让 CPU 不必等待拷贝完成
        self.mailbox = self.mailbox.to(ctx_device, non_blocking=True)
        self.mailbox_ts = self.mailbox_ts.to(ctx_device, non_blocking=True)
        self.next_mail_pos = self.next_mail_pos.to(ctx_device, non_blocking=True)
        self.device = ctx_device

    def set_mailbox_local(self, index, source, source_ts, Reduce_Op=None):
        # 确保数据搬运与计算异步进行
        index = index.to(self.device, non_blocking=True)
        source = source.to(self.device, non_blocking=True)
        source_ts = source_ts.to(self.device, non_blocking=True)

        if Reduce_Op == 'max':
            unq_id, inv = index.unique(return_inverse=True)
            max_ts, idx = torch_scatter.scatter_max(source_ts, inv, dim=0)
            source_ts = max_ts
            source = source[idx]
            index = unq_id

        # 写入 Pinned Memory 的速度比普通内存更快
        pos = self.next_mail_pos[index]
        self.mailbox_ts[index, pos] = source_ts
        self.mailbox[index, pos] = source
        
        if self.mailbox_size > 1:
            self.next_mail_pos[index] = torch.remainder(pos + 1, self.mailbox_size)

    def get_message(self, idx):
        """
        高性能读取：
        如果 idx 在 GPU 而 mailbox 在 CPU (Pinned)，则触发 UVA 零拷贝读取。
        如果 mailbox 已经在 GPU，则是极速的显存内访问。
        """
        return self.mailbox[idx], self.mailbox_ts[idx]

class DistMailbox:
    def __init__(self, state: PartitionState, mailbox: Mailbox):
        self.state = state
        self.mailbox = mailbox
        self.node_mapper = state.node_mapper
    @property
    def device(self):
        return self.mailbox.device
    def get_mail_by_gid(self, gids: Union[Tensor, DistRouteIndex]) -> Tuple[Tensor, Tensor]:
        target_device = self.mailbox.device
        
        # 1. 快速路径：DistRouteIndex 位运算解包
        if isinstance(gids, DistRouteIndex):
            loc_indices = gids.loc.to(target_device, non_blocking=True)
        else:
            # 2. 慢速路径：标准 ID 转换
            if gids.device != target_device:
                gids = gids.to(target_device, non_blocking=True)
            loc_indices = self.node_mapper.to_local(gids, device=target_device)
            
        return self.mailbox.get_message(loc_indices)

    def set_mail_by_gid(self, gids: Tensor, source: Tensor, source_ts: Tensor, Reduce_Op: str = 'max'):
        target_device = self.mailbox.device
        
        # 异步传输
        gids = gids.to(target_device, non_blocking=True)
        source = source.to(target_device, non_blocking=True)
        source_ts = source_ts.to(target_device, non_blocking=True)
        
        loc_indices = self.node_mapper.to_local(gids, device=target_device)
        
        # 过滤本地存在的节点
        mask = loc_indices >= 0
        if mask.any():
            self.mailbox.set_mailbox_local(
                loc_indices[mask], 
                source[mask], 
                source_ts[mask], 
                Reduce_Op=Reduce_Op
            )
         
    def update(self, idx:torch.Tensor|Tuple[torch.Tensor,...],
                    efeat:torch.Tensor,
                    upd_mem:torch.Tensor| Tuple[torch.Tensor,...], 
                    upd_ts:torch.Tensor,
                    Reduce_Op: str = 'max'):
        if(isinstance(idx, tuple) and isinstance(upd_mem, tuple)):
            idx = torch.cat(idx, dim=0)
            num_edge = idx.shape[0] // 2
            src_mem, dst_mem = upd_mem
        else:
            num_edge = idx.shape[0] // 2
            src_mem = upd_mem[:num_edge]
            dst_mem = upd_mem[num_edge:]
        if efeat is None:
            efeat = torch.zeros((idx.shape[0], 0), dtype=src_mem.dtype, device=src_mem.device)
        src_message = torch.cat([src_mem, dst_mem, efeat.to(src_mem.device)], dim=1)
        dst_message = torch.cat([dst_mem, src_mem, efeat.to(dst_mem.device) ], dim=1)
        messages = torch.cat([src_message, dst_message], dim=0)
        timestamps = torch.cat([upd_ts, upd_ts], dim=0)
        if Reduce_Op == 'max':
            unq_id,inv = idx.unique(return_inverse = True)
            max_ts,id =  torch_scatter.scatter_max(timestamps.to(self.device),inv.to(self.device),dim=0)
            timestamps = max_ts
            messages = messages[id.to(self.device)]
            idx = unq_id
        self.set_mail_by_gid(
            gids=idx,
            source=messages,
            source_ts=timestamps,
            Reduce_Op=Reduce_Op
        )
    def generate_message_and_update(self, task, new_mem, new_mem_ts, graph_context):
        num_src = task['task_src'].shape[0]
        new_mem_inv = task['mem_inv_map']
        src_mem = new_mem[new_mem_inv[:num_src]]
        dst_mem = new_mem[new_mem_inv[num_src:2*num_src]]
        upd_ts = task['task_ts']
        self.update(
            (task['task_src'],task['task_dst']),
            efeat=graph_context.edge_feats[task['task_eid']],
            upd_mem = (src_mem, dst_mem),
            upd_ts=upd_ts
        )
    @property
    def shape(self):
        return self.mailbox.shape