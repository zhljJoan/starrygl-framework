import torch
import dgl
import queue
import threading
from typing import Optional, Any, List, Union, Tuple
from starrygl.data.structs import StarryBatchData

class HostToDevicePrefetcher:
    def __init__(self, 
                 loader, 
                 device: torch.device, 
                 partition_state, 
                 context: Any,  
                 queue_size: int = 3,
                 hist_cache = None):
        self.loader = loader
        self.device = device
        self.state = partition_state
        self.context = context
        self.hist_cache = hist_cache    
        
        # 传输专用流：优先级设为高，确保 H2D 尽快执行
        self.stream = torch.cuda.Stream(device=device, priority=-1)
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.worker_thread = None

    def _worker_loop(self):
        """后台线程：负责从 CPU DataLoader 中不断提取 Batch"""
        try:
            for batch in self.loader:
                if self.stop_event.is_set():
                    break
                self.queue.put(batch)
            # 放入 None 表示数据流结束
            self.queue.put(None)
        except Exception as e:
            print(f"[Prefetcher] Worker loop error: {e}")
            self.queue.put(None)

    def _shutdown(self):
        """清理线程和队列"""
        self.stop_event.set()
        # 清空队列以释放阻塞的生产者线程
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)

    def __iter__(self):
        """标准的迭代协议实现"""
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        # 预热 Pipeline：获取第一个 Batch
        first_batch = self.queue.get()
        if first_batch is None:
            self._shutdown()
            return

        # 异步提交第一个任务到显存
        next_future_bundle = self.preload(first_batch)

        while True:
            current_batch = first_batch
            current_bundle = next_future_bundle
            
            # 提前从队列获取下一个 CPU Batch
            next_batch_cpu = self.queue.get()
            
            if next_batch_cpu is None:
                # 最后一个 Batch 的组装并产出
                yield self.assemble(current_batch, current_bundle)
                break
                
            # 异步提交下一个传输任务，同时 yield 当前已准备好的数据
            next_future_bundle = self.preload(next_batch_cpu)
            yield self.assemble(current_batch, current_bundle)
            first_batch = next_batch_cpu

        self._shutdown()
    
    def preload(self, batch):
        if batch is None: return None
        
        # CPU 索引阶段
        input_gids_cpu = batch.layer_data[-1]['gids']
        
        bundle = {
            'node_feat_future': None,
            'mail_future': None,
            'hist_futures': [],
            'edge_futures': [],
            'struct_event': None
        }

        with torch.cuda.stream(self.stream):
            # A. Node Features (UVA Prefetch)
            if self.context.node_feats:
                bundle['node_feat_future'] = self.context.node_feats.prefetch(
                    input_gids_cpu, stream=self.stream
                )

            # B. Mailbox (UVA Prefetch)
            if self.context.mailbox:
                # 修正：loc_idx 索引直接用于切片，利用 pinned memory UVA
                loc_idx = self.state.node_mapper.to_local(input_gids_cpu, device=torch.device('cpu'))
                
                # 异步拷贝到 GPU
                mail_gpu = self.context.mailbox.mailbox.mailbox[loc_idx].to(self.device, non_blocking=True)
                ts_gpu = self.context.mailbox.mailbox.mailbox_ts[loc_idx].to(self.device, non_blocking=True)
                bundle['mail_future'] = (mail_gpu, ts_gpu)

            # C. 逐层特征
            pre_gids = batch.task_data['gids'] if 'gids' in batch.task_data else None
            for i, layer in enumerate(batch.layer_data):
                # Edge
                if self.context.edge_feats and layer.get('eids') is not None:
                    eids = layer['eids']
                    ef_gpu = self.context.edge_feats.features[eids].to(self.device, non_blocking=True)
                    bundle['edge_futures'].append(ef_gpu)
                else:
                    bundle['edge_futures'].append(None)

                # History
                if self.hist_cache and layer.get('gids') is not None:
                    h_gpu = self.hist_cache.node_states[i].prefetch(pre_gids, stream=self.stream)
                    bundle['hist_futures'].append(h_gpu)
                    pre_gids = layer['gids']
            if len(self.hist_cache.node_states) > len(batch.layer_data):
                h_gpu = self.hist_cache.node_states[-1].prefetch(input_gids_cpu, stream=self.stream)
                bundle['hist_futures'].append(h_gpu)
            # D. 结构传输：这一步必须在所有 CPU 索引计算完成后执行
            batch.to(self.device, non_blocking=True)
            
            event = torch.cuda.Event()
            event.record(self.stream)
            bundle['struct_event'] = event

        return bundle

    def assemble(self, batch, bundle) -> StarryBatchData:
        if batch is None: return None
        
        # 等待数据/结构流同步
        if bundle['struct_event']:
            bundle['struct_event'].wait()

        mfgs = batch.blocks
        
        if bundle['node_feat_future'] is not None:
            # 如果 node_feat_future 是 tuple (feat, ts)，取第一个
            data = bundle['node_feat_future']
            mfgs[0].srcdata['x'] = data[0] if isinstance(data, tuple) else data

        # 修正引用：is_master_by_gid 应通过 context 或 state 调用
        input_gids_gpu = mfgs[0].srcdata[dgl.NID]
        # 优化：通过 PartitionState 直接判断是否为本地 Master
        is_remote = ~self.state.node_mapper.is_master_by_local(
                        self.state.node_mapper.to_local(input_gids_gpu, device=self.device)
                    )

        task = batch.task_data
        pos = (task.get('task_src'), task.get('task_dst'), task.get('ts')) if 'task_src' in task \
              else (task.get('task_node'), task.get('ts'))

        return StarryBatchData(
            mfgs=mfgs,
            dist_flag=is_remote,
            history=bundle['hist_futures'],
            mailbox=bundle['mail_future'],
            nid_mapper=batch.layer_data[-1]['gids'], # 保留原始 GID 以便反查
            roots=(pos, task.get('neg_pool')),
            routes=batch.comm_plans
        )