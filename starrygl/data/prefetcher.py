import torch
import dgl
import queue
import threading
from typing import Optional, Any, List, Union
from starrygl.data.structs import StarryBatchData
from starrygl.utils.async_io import AsyncTransferFuture

class HostToDevicePrefetcher:
    def __init__(self, 
                 loader, 
                 device: torch.device, 
                 partition_state, 
                 mailbox_wrapper: Any = None,       
                 history_wrapper: Union[Any, List[Any]] = None,       
                 node_feat_cpu: Optional[torch.Tensor] = None,
                 edge_feat_cpu: Optional[torch.Tensor] = None,
                 queue_size: int = 3
                 ):
        self.loader = loader
        self.device = device
        self.partition_state = partition_state
        self.mailbox = mailbox_wrapper
        self.history = history_wrapper
        self.node_feat = node_feat_cpu
        self.edge_feat = edge_feat_cpu
        
        # 传输专用流
        self.stream = torch.cuda.Stream(priority=-1)
        
        # 线程队列
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.worker_thread = None

    def _worker_loop(self):
        try:
            iter_loader = iter(self.loader)
            for batch in iter_loader:
                if self.stop_event.is_set(): break
                self.queue.put(batch)
            self.queue.put(None)
        except Exception as e:
            print(f"[Prefetcher] Thread Error: {e}")
            self.queue.put(None)

    def __iter__(self):
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        # 启动 Pipeline
        first_batch = self.queue.get()
        if first_batch is None:
            self._shutdown()
            return

        # 提交第一个任务 (异步)
        next_future_bundle = self.preload(first_batch)

        while True:
            current_batch = first_batch
            current_bundle = next_future_bundle
            
            next_batch_cpu = self.queue.get()
            
            if next_batch_cpu is None:
                yield self.assemble(current_batch, current_bundle)
                break
                
            next_future_bundle = self.preload(next_batch_cpu)
            yield self.assemble(current_batch, current_bundle)
            first_batch = next_batch_cpu

        self._shutdown()

    def _shutdown(self):
        self.stop_event.set()
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except: pass
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)

    def preload(self, batch):
        """
        [High-Performance Prefetch]
        策略: 
        1. 先提取所有 CPU 索引 (Avoid device mismatch error).
        2. 先提交 Feature 传输 (Maximize PCIe throughput).
        3. 最后移动 Batch 结构 (Overlapping).
        """
        if batch is None: return None
        
        # =========================================================
        # 1. [CPU Phase] 贪婪提取索引 (必须在 batch.to 之前完成)
        # =========================================================
        # 提取 Input Layer GIDs (用于 Node Feat / Mailbox)
        input_nids_cpu = batch.layer_data[-1]['gids']
        
        # 预先提取每一层的 EIDs 和 GIDs (用于 Edge Feat / History)
        # 我们把这些索引保存在 list 中，供稍后切片使用
        layer_indices_cpu = []
        for layer in batch.layer_data:
            # 只要引用，不发生拷贝，速度极快
            eids = layer.get('eids') 
            gids = layer.get('gids')
            layer_indices_cpu.append((eids, gids))

        # =========================================================
        # 2. [GPU Stream Phase] 异步提交任务
        # =========================================================
        bundle = {
            'edge_futures': [], 
            'hist_futures': [],
            'node_feat_future': None,
            'struct_event': None,
            'mail_feat_future': None,
            'mail_ts_future': None
        }

        with torch.cuda.stream(self.stream):
            
            # --- A. Node Features (通常数据量最大，最先提交) ---
            if self.node_feat is not None:
                # [关键]: input_nids_cpu 是 CPU Tensor，node_feat 是 CPU Tensor
                # 如果 node_feat 是 pin_memory 的，这步是纯异步的
                nf_chunk = self.node_feat[input_nids_cpu] 
                
                nf_gpu = nf_chunk.to(self.device, non_blocking=True)
                
                event = torch.cuda.Event()
                event.record(self.stream)
                bundle['node_feat_future'] = AsyncTransferFuture(nf_gpu, event)

            # --- B. Edge Features & History (逐层提交) ---
            for i, (eids_cpu, gids_cpu) in enumerate(layer_indices_cpu):
                # 1. Edge Feat
                e_future = None
                if self.edge_feat is not None and eids_cpu is not None and eids_cpu.numel() > 0:
                    # 使用刚才保存的 CPU 索引切片
                    ef_chunk = self.edge_feat[eids_cpu]
                    
                    ef_gpu = ef_chunk.to(self.device, non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(self.stream)
                    e_future = AsyncTransferFuture(ef_gpu, event)
                bundle['edge_futures'].append(e_future)

                # 2. History
                h_future = None
                if isinstance(self.history, list) and i < len(self.history):
                    # History 获取通常涉及 hash map 查找，可能略慢，放在后面
                    # 假设 .get() 返回的是 CPU Tensor (如果是 GPU Tensor 则不需要 .to)
                    h_data = self.history[i].get(gids_cpu) 
                    if isinstance(h_data, tuple):
                        h_emb, h_ts = h_data
                        h_gpu = torch.cat([h_emb, h_ts], dim=1).to(self.device, non_blocking=True)
                    else:
                        h_gpu = h_data.to(self.device, non_blocking=True)
                        
                    event = torch.cuda.Event()
                    event.record(self.stream)
                    h_future = AsyncTransferFuture(h_gpu, event)
                bundle['hist_futures'].append(h_future)

            # --- C. Mailbox (涉及 Partition 查找) ---
            if self.mailbox is not None:
                # 处理分布式 ID 映射
                if hasattr(self.partition_state, 'to_local_nid_cpu'):
                    input_local_nids = self.partition_state.to_local_nid_cpu(input_nids_cpu)
                else:
                    input_local_nids = input_nids_cpu
                
                mb_feat, mb_ts = self.mailbox.get_message(input_local_nids)

                mb_feat_gpu = mb_feat.to(self.device, non_blocking=True)
                mb_ts_gpu = mb_ts.to(self.device, non_blocking=True)

                event = torch.cuda.Event()
                event.record(self.stream)
                bundle['mail_feat_future'] = AsyncTransferFuture(mb_feat_gpu, event)
                bundle['mail_ts_future'] = AsyncTransferFuture(mb_ts_gpu, event)

            # --- D. 最后移动 Batch 结构 ---
            # 放到最后是因为其他切片操作依赖于 batch 在 CPU 上
            # 一旦执行这一步，batch 内部的 layer_data 里的 tensor 都会变成 GPU Tensor
            batch.to(self.device, non_blocking=True)
            
            # 记录结构传输完成
            struct_event = torch.cuda.Event()
            struct_event.record(self.stream)
            bundle['struct_event'] = struct_event

        return bundle

    def assemble(self, batch, bundle) -> StarryBatchData:
        if batch is None: return None
        
        # 1. 确保结构到位
        # 虽然 Assemble 本身是在 CPU 跑，但 batch.blocks 已经是 GPU 对象
        # 我们在这里做一个轻量级的 wait，确保 DGL 能读到图结构
        if bundle['struct_event']:
            bundle['struct_event'].wait() # 让当前流等待结构传输完成
        
        mfgs = batch.blocks
        if bundle['node_feat_future']:
            mfgs[0].srcdata['x'] = bundle['node_feat_future'].wait()
            
        # 3. 注入 Edge Feat Future
        e_futures = bundle.get('edge_futures', [])
        for i, future in enumerate(e_futures):
            if i < len(mfgs) and future is not None:
                mfgs[i].edata['e_handle'] = future
                
        # 4. 注入 History Future
        h_futures = bundle.get('hist_futures', [])     
        mailbox_list = None
        if 'mail_feat_future' in bundle:
            mailbox_list = [
                bundle['mail_feat_future'].wait(), 
                bundle['mail_ts_future'].wait()
            ]

        global_nids = mfgs[0].srcdata[dgl.NID]
        
        if hasattr(self.partition_state, 'loc_ids'):
             is_local = torch.isin(global_nids, self.partition_state.loc_ids)
             is_remote = ~is_local
        else:
             is_remote = torch.zeros_like(global_nids, dtype=torch.bool)
             
        dist_flags = [is_remote] + [None] * (len(mfgs) - 1)
        for i in range(1, len(mfgs)):
            if hasattr(mfgs[i], 'dstdata'):
                mfgs[i].dstdata['is_remote'] = is_remote[mfgs[i].dstdata[dgl.NID]]
            dist_flags[i] = is_remote[mfgs[i].srcdata[dgl.NID]]
       
        task = batch.task_data
        if 'task_src' in task:
            pos = (task['task_src'], task['task_dst'], task['ts'], )
        else:
            pos = (task.get('task_node'),task['ts'])
            
        neg = task.get('neg_pool') # 负样本，可能为 None

        # 7. 返回完整对象
        return StarryBatchData(
            mfgs=mfgs,                  # DGL Blocks (GPU, 含 Edge/Node Future Handle)
            dist_flag=dist_flags,       # Halo Mask (GPU Bool Tensor)
            history=h_futures,          # History Futures List (PCIe 传输中)
            mailbox=mailbox_list,       # Mailbox Futures List (PCIe 传输中)
            nid_mapper=batch.layer_data[-1]['gids'], 
            roots=(pos, neg),           
            routes=batch.comm_plans    
        )