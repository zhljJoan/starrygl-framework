import torch
import dgl
import queue
import threading
from typing import Optional, Any, List, Union, Tuple
from starrygl.data.structs import StarryBatchData
# 假设 AsyncTransferFuture 内部封装了 tensor 和 event

class HostToDevicePrefetcher:
    def __init__(self, 
                 loader, 
                 device: torch.device, 
                 partition_state, 
                 context: Any,  # 传入 StarryglGraphContext 聚合对象
                 queue_size: int = 3):
        self.loader = loader
        self.device = device
        self.state = partition_state
        self.context = context
        
        # 传输专用流
        self.stream = torch.cuda.Stream(device=device, priority=-1)
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.worker_thread = None

    def preload(self, batch):
        if batch is None: return None
        
        # 1. [CPU Phase] 获取最外层（Input Layer）的全局 GIDs
        input_gids_cpu = batch.layer_data[-1]['gids']
        
        bundle = {
            'node_feat_future': None,
            'mail_future': None,
            'hist_futures': [],
            'edge_futures': [],
            'struct_event': None
        }

        # 开启高性能流
        with torch.cuda.stream(self.stream):
            # --- A. Node Features (UVA Prefetch) ---
            if self.context.node_feats:
                # 利用我们在 DistFeatureCache 中定义的 prefetch 逻辑
                # 内部会自动处理 pin_memory 和 non_blocking=True
                bundle['node_feat_future'] = self.context.node_feats.prefetch(
                    input_gids_cpu, stream=self.stream
                )

            # --- B. Mailbox (UVA Prefetch) ---
            if self.context.dist_mailbox:
                # 获取本地索引并直接切片传输
                loc_idx = self.state.node_mapper.to_local(input_gids_cpu, device=torch.device('cpu'))
                
                mail_gpu = self.context.dist_mailbox.mailbox.mailbox[loc_idx].to(self.device, non_blocking=True)
                ts_gpu = self.context.dist_mailbox.mailbox.mailbox_ts[loc_idx].to(self.device, non_blocking=True)
                bundle['mail_future'] = (mail_gpu, ts_gpu)

            # --- C. 逐层处理 Edge 和 History ---
            for i, layer in enumerate(batch.layer_data):
                # Edge Features
                if self.context.edge_feats and layer.get('eids') is not None:
                    eids = layer['eids']
                    ef_gpu = self.context.edge_feats.features[eids].to(self.device, non_blocking=True)
                    bundle['edge_futures'].append(ef_gpu)
                else:
                    bundle['edge_futures'].append(None)

                # History (Node State)
                if self.context.node_state and layer.get('gids') is not None:
                    # 针对 DTDG 场景的高性能获取
                    h_gpu = self.context.node_state.prefetch(layer['gids'], stream=self.stream)
                    bundle['hist_futures'].append(h_gpu)

            # --- D. 异步移动图结构 (DGL Blocks) ---
            # 必须在所有 CPU 侧索引切片完成后执行
            batch.to(self.device, non_blocking=True)
            
            event = torch.cuda.Event()
            event.record(self.stream)
            bundle['struct_event'] = event

        return bundle

    def assemble(self, batch, bundle) -> StarryBatchData:
        if batch is None: return None
        
        # 等待结构流同步，确保 DGL Block 已经完全进入显存
        if bundle['struct_event']:
            bundle['struct_event'].wait()

        mfgs = batch.blocks
        
        # 注入 Node Features
        if bundle['node_feat_future'] is not None:
            # 这里注入的是已经开始往显存搬运的 Tensor
            mfgs[0].srcdata['x'] = bundle['node_feat_future']

        # 离散时间动态图通常需要注入分布式掩码
        # 利用 bit manipulation 检查 Master/Replica 状态
        input_gids_gpu = mfgs[0].srcdata[dgl.NID]
        # 假设 DistRouteIndex(input_gids_gpu).is_shared 是我们要的 flag
        # 这里性能最高，因为不需要查表，直接位运算
        is_remote = self.nid_mapper.is_master_by_gid(input_gids_gpu) 

        # 构建任务元数据
        task = batch.task_data
        pos = (task.get('task_src'), task.get('task_dst'), task.get('ts')) if 'task_src' in task \
              else (task.get('task_node'), task.get('ts'))

        return StarryBatchData(
            mfgs=mfgs,
            dist_flag=is_remote,
            history=bundle['hist_futures'],
            mailbox=bundle['mail_future'],
            nid_mapper=batch.layer_data[-1]['gids'],
            roots=(pos, task.get('neg_pool')),
            routes=batch.comm_plans
        )