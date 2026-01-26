import torch
import dgl
import queue
import threading
from typing import Optional, Any, List, Union, Tuple
from starrygl.data.graph_context import StarryglGraphContext
from starrygl.data.structs import StarryBatchData
from starrygl.utils.async_io import AsyncTransferFuture
from starrygl.utils.partition_book import PartitionState

class HostToDevicePrefetcher:
    def __init__(self, 
                 loader, 
                 device: torch.device, 
                 partition_state: PartitionState, 
                 context: StarryglGraphContext,  
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
        
        num_layers = len(batch.layer_data)
        input_gids_cpu = batch.layer_data[-1]['gids']
        
        # 预分配占位符
        bundle = {
            'node_feat_future': None,
            'mail_future': None,
            'hist_futures': [None] * len(self.hist_cache.node_states) if self.hist_cache is not None else None, # 增加一个位置给 Input Layer
            'edge_futures': [None] * num_layers,
            'struct_event': None,
            'is_remote': [None] * len(self.hist_cache.node_states) if self.hist_cache is not None else None
        }

        with torch.cuda.stream(self.stream):
            # 提取每一层的 eid 和用于 history 的 gids (保持在 CPU)
            layer_eids_cpu = [layer.get('eid') for layer in batch.layer_data]
            layer_gids_cpu = [layer.get('gids') for layer in batch.layer_data]
            task_gids_cpu = batch.task_data.get('gids')
            batch.to(self.device, non_blocking=True)
            event = torch.cuda.Event(); event.record(self.stream)
            bundle['struct_event'] = event
            # 1. 最先发送：输入层基础特征（模型计算的第一步）
            if self.context.node_feats:
                nf_gpu = self.context.node_feats[input_gids_cpu].to(self.device, non_blocking=True)
                ev = torch.cuda.Event(); ev.record(self.stream)
                bundle['node_feat_future'] = AsyncTransferFuture(nf_gpu, ev)

            if self.context.mailbox:
                mb, mb_ts = self.context.mailbox.get_mail_by_gid(input_gids_cpu)
                mb = mb.to(self.device, non_blocking=True)
                mb_ts = mb_ts.to(self.device, non_blocking=True)
                ev = torch.cuda.Event(); ev.record(self.stream)
                bundle['mail_future'] = AsyncTransferFuture((mb, mb_ts), ev)
            # 2. 倒序发送：从 Layer N (输入侧) 往 Layer 0 (输出侧) 发送
            # 这样保证了 GPU 任务队列里，最先被 wait 的数据最先到达
            for l, i in enumerate(range(num_layers - 1, -1, -1)):
                #print(input_gids_cpu.shape , layer_gids_cpu[i].shape)
                #layer = batch.layer_data[i]
                curr_eid = layer_eids_cpu[i]#self.state.edge_mapper.to_local(layer_eids_cpu[i], device=torch.device('cpu'))
                if self.context.edge_feats and curr_eid is not None:
                    ef_gpu = self.context.edge_feats[curr_eid].to(self.device, non_blocking=True)
                    ev = torch.cuda.Event(); ev.record(self.stream)
                    bundle['edge_futures'][l] = AsyncTransferFuture(ef_gpu, ev)

                pre_gids = layer_gids_cpu[i] if i >= 0 else task_gids_cpu
                if self.hist_cache and pre_gids is not None:
                    loc = self.state.node_mapper.to_local(pre_gids, device=torch.device('cpu'))
                    h_gpu,h_ts = self.hist_cache.node_states[i][loc]
                    h_gpu = h_gpu.to(self.device, non_blocking=True)
                    h_ts = h_ts.to(self.device, non_blocking=True)
                    bundle['is_remote'][l] = (~self.state.node_mapper.is_master_by_local(loc)).to(self.device, non_blocking=True)
                    ev = torch.cuda.Event(); ev.record(self.stream)
                    bundle['hist_futures'][l] = AsyncTransferFuture((h_gpu, h_ts), ev)

            # 3. 发送最底层的初始 Node State
            if self.hist_cache and len(self.hist_cache.node_states) > num_layers:
                loc = self.state.node_mapper.to_local(task_gids_cpu, device=torch.device('cpu'))
                h_gpu,h_ts = self.hist_cache.node_states[-1][loc]
                h_gpu = h_gpu.to(self.device, non_blocking=True)
                h_ts = h_ts.to(self.device, non_blocking=True)
                ev = torch.cuda.Event(); ev.record(self.stream)
                bundle['hist_futures'][-1] = AsyncTransferFuture((h_gpu, h_ts), ev)


        return bundle

    def assemble(self, batch, bundle) -> StarryBatchData:
        if batch is None: return None
        
        # 仅等待结构，不阻塞计算流
        if bundle['struct_event']:
            torch.cuda.current_stream().wait_event(bundle['struct_event'])
        mfgs = batch.blocks
        num_blocks = len(mfgs)
        
        for i in range(num_blocks):
            # 注入 Future，不调用 wait()
            if i < len(bundle['edge_futures']):
                mfgs[i].e_future = bundle['edge_futures'][i]
            
            if i < len(bundle['hist_futures']):
                mfgs[i].h_future = bundle['hist_futures'][i]
                mfgs[i].is_remote = bundle['is_remote'][i]

        if bundle['node_feat_future']:
            mfgs[0].x_future = bundle['node_feat_future']
        #print(batch.comm_plans, batch.comm_plans[:len(bundle['hist_futures'])], batch.comm_plans[:len(bundle['hist_futures'])].reverse())
        comm_plan = batch.comm_plans[:len(bundle['hist_futures'])]
        comm_plan.reverse()
        return StarryBatchData(
            mfgs=mfgs,
            history = None,
            mailbox = bundle['mail_future'],
            nid_mapper=None,
            roots=batch.task_data,
            routes=comm_plan
        )