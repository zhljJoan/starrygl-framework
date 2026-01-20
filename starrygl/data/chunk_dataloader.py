from mailbox import Mailbox
import torch
import torch.distributed as dist
import numpy as np
import dgl
import threading
import queue
import math
import random
import collections
from collections import deque
from typing import List, Optional, Iterator, Tuple, Dict, Any
from dataclasses import dataclass
from dgl.heterograph import DGLBlock
# 假设的内部模块引用
from starrygl.graph.graph import AsyncBlock, AsyncGraphBlob, pyGraph
from starrygl.data.partitions import PartitionData
from starrygl.cache.NodeState import NodeState
from starrygl.route.route import PartitionState
from starrygl.utils.context import DistributedContext
from numba import njit



class NegativeSampler:
    def __init__(self, num_nodes: int, strategy: str = 'uniform'):
        self.num_nodes = num_nodes
        self.strategy = strategy
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        随机采样 num_samples 个节点作为负样本。
        """
        if self.strategy == 'uniform':
            return torch.randint(0, self.num_nodes, (num_samples,), dtype=torch.long)
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not implemented")
        
class NegativeSetSampler:
    def __init__(self, nodes: torch.Tensor, strategy: str = 'uniform'):
        self.nodes = nodes
        self.strategy = strategy
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        随机采样 num_samples 个节点作为负样本。
        """
        if self.strategy == 'uniform':
            return self.nodes[torch.randint(0, len(self.nodes), (num_samples,), dtype=torch.long)]
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not implemented")        
        

def cal_boundaries_rolling(ts, batch_size, max_span = 10000):
    """
    滚动计算边界，确保每个 Batch 至少有 batch_size 个事件。
    解决了 "时间戳重叠导致下一个 Batch 变小" 的问题。
    """
    # 1. 计算前缀和 (Cumulative Sum)
    # cum_counts[i] 表示第 i 个时间点结束时，累计有多少事件
    ts_min = torch.tensor([ts[0]], device=ts.device, dtype=ts.dtype)
    dist.all_reduce(ts_min, op=dist.ReduceOp.MIN)
    unique_ts, counts = torch.unique(ts, return_counts=True)
    cum_counts = torch.cumsum(counts, dim=0)
    n_unique = len(cum_counts)
    boundaries = [int(ts_min.item())]
    last_cum_count = 0
    total_events = cum_counts[-1]
    
    while last_cum_count < total_events:
        target = last_cum_count + batch_size
        #print(last_cum_count, target, total_events)
        if target >= total_events:
            if boundaries[-1] != unique_ts[-1] + 1:
                boundaries.append(unique_ts[-1] + 1)
            break
        idx = torch.searchsorted(cum_counts, target, side='left')

        boundaries.append(unique_ts[idx])
        last_cum_count = cum_counts[idx]
    #print(last_cum_count)
    boundaries = torch.tensor(boundaries, dtype=torch.long, device =DistributedContext.get_default_context().device)
    #print('finish rank {} {} {}'.format(torch.distributed.get_rank(),len(boundaries),boundaries))
    torch.distributed.all_reduce(boundaries, op=torch.distributed.ReduceOp.MAX) 
    #print('all reduce {}'.format(boundaries))
    return boundaries


@dataclass
class StarryBatchData:
    """标准化的 Batch 返回结构"""
    mfgs: List[DGLBlock]
    dist_flag: List[torch.Tensor]
    history: List[NodeState]
    mailbox: Mailbox
    nid_mapper: torch.Tensor # 用于映射本地 ID 到全局 ID
    roots: Tuple[torch.Tensor, torch.Tensor] # (正样本 ID, 负样本 ID)
    def __init__(self,mfgs: List[DGLBlock], dist_flag: List[torch.Tensor], history: List[NodeState],
                 mailbox: Mailbox, nid_mapper: torch.Tensor, roots: Tuple[torch.Tensor, torch.Tensor]):
        self.mfgs = mfgs
        self.dist_flag = dist_flag
        self.history = history
        self.mailbox = mailbox
        self.nid_mapper = nid_mapper
        self.roots = roots


class StarryGLDataset:
    def __init__(self, 
                 node_chunk_id: torch.Tensor, 
                 ts: torch.Tensor, 
                 event_chunk_id: torch.Tensor = None, # [必须] 事件对应的 Chunk ID
                 rank: int = 0, 
                 world_size: int = 1,
                 local_chunks = None):
        self.ts = ts
        self.rank = rank
        self.world_size = world_size
        
        # 1. 预处理 Event Chunk ID
        # 如果没有直接提供，尝试从 node_chunk_id 推导 (需要 dst_nid，此处假设已传入)
        # 这里的 event_chunk_id 必须与 ts 长度一致且一一对应
        if event_chunk_id is None:
            raise ValueError("Smart strategy requires 'event_chunk_id' mapping edges to chunks.")
            
        # 确保在 CPU 上以便进行快速的切片和 unique 操作
        if event_chunk_id.device.type != 'cpu':
            self.event_chunk_id = event_chunk_id.cpu()
        else:
            self.event_chunk_id = event_chunk_id
            
        # ... (其他初始化保持不变)
        self.min_t = torch.tensor([self.ts.min().item()])
        self.max_t = torch.tensor([self.ts.max().item()])
        
    def _pack_chunks_sequentially(self, chunk_ids: List[int], event_counts: List[int], 
                              max_events: int) -> List[Tuple[List[int], int]]:
        packed_batches = []
        
        current_ids = []
        current_load = 0
        
        for c_id, count in zip(chunk_ids, event_counts):
            # 检查是否需要封包 (当前包非空 且 (加进来会超载 或 Chunk数已满))
            is_full_events = (current_load + count > max_events) and (len(current_ids) > 0)
            if is_full_events:
                packed_batches.append((current_ids, current_load))
                current_ids = []
                current_load = 0
            
            current_ids.append(c_id)
            current_load += count
            
        # 处理最后一个包
        if current_ids:
            packed_batches.append((current_ids, current_load))
            
        return packed_batches
    
    def generate_tasks_smart(
        self,
        batch_size: int,         # 目标 Batch 大小 (事件数)
        sub_batch_size: int = 2000, # [空间维度] 实际 Task 的目标事件数 (装箱限制)
        max_span: int = 10000000,
        negative_sampler: Optional[NegativeSampler] = None # 传入负采样器实例
    ) -> Iterator[dict]:

        # 1. 准备数据
        # if isinstance(self.ts, torch.Tensor):
        #     ts_np = self.ts.cpu().numpy()
        # else:
        #     ts_np = self.ts
        self.ts = self.ts.to(DistributedContext.get_default_context().device)
        boundaries = cal_boundaries_rolling(self.ts, batch_size, max_span)
        print(self.ts.device, boundaries.device)
        boundary_indices = torch.searchsorted(self.ts, boundaries, right = False)
        print('{}\n'.format( boundary_indices))
        curr_idx_start = 0
        step = 0
        for i,idx_end in enumerate(boundary_indices[1:]):
            t_start = boundaries[i-1]
            t_end = boundaries[i]
            print(t_start, t_end, idx_end.item(), boundary_indices[i].item())
            idx_start = boundary_indices[i]
            print('{} {} {}\n'.format(i,curr_idx_start, idx_start.item()))
            window_chunk_ids = self.event_chunk_id[idx_start:idx_end]
            active_local_chunks, active_counts = torch.unique(window_chunk_ids, return_counts=True, sorted=True)
            active_chunks_list = active_local_chunks.tolist()
            active_counts_list = active_counts.tolist()
            
            # 快速查找表 (用于后续构建 sub_meta)çç
            task_meta_lookup = dict(zip(active_chunks_list, active_counts_list))
            
            # === D. 执行顺序装箱 (Bin Packing) ===
            # 将活跃 chunk 组合成若干个负载均衡的 sub-batch
            packed_batches = self._pack_chunks_sequentially(
                active_chunks_list,
                active_counts_list,
                max_events=sub_batch_size,   # 核心：控制显存占用
            )
            
            # 懒加载：只有在确实需要时间戳时才去切片原始 ts
            # 避免对整个 window_ts 进行切片
            window_ts_view = self.ts[curr_idx_start : idx_end]
            
            for sub_step, (chunk_subset_ids, expected_sub_size) in enumerate(packed_batches):
                chunk_subset = torch.tensor(chunk_subset_ids, dtype=torch.long)
                mask = torch.isin(window_chunk_ids, chunk_subset)
                sub_batch_ts = window_ts_view[mask]
                sub_meta = {cid: task_meta_lookup[cid] for cid in chunk_subset_ids}
                neg_roots = None
                neg_roots_ts = None
                if negative_sampler is not None and expected_sub_size > 0:
                    neg_roots = negative_sampler.sample(expected_sub_size)
                    neg_roots_ts = sub_batch_ts.clone()
                
                yield {
                    'time_start': t_start.item(),
                    'time_end': t_end.item(),
                    'chunks': chunk_subset,      # Tensor: 本批次涉及的 chunks
                    'step': step,
                    'sub_step': sub_step,
                    'expected_size': expected_sub_size, # 准确的事件数 (用于分配显存)
                    'meta': sub_meta,            # 详细的 {chunk: count}
                    'neg_roots': neg_roots,      
                    'neg_roots_ts': neg_roots_ts 
                }
            
            # 更新全局指针
            curr_idx_start = idx_end
            step += 1
             
            
        

class GraphCollator:
    def __init__(self, 
                 partition_data: PartitionData, 
                 history_state: NodeState,     # Historical Embedding (List or Single)
                 mailbox: Mailbox,                 # Mailbox 对象
                 partition_state: PartitionState,
                 self_loop: bool = True,
                 add_inverse_edge:bool = False,
                 device: torch.device = None):
        self.partition_data = partition_data
        self.history_state = history_state
        self.mailbox = mailbox
        self.partition_state = partition_state
        self.device = DistributedContext.get_default_context().device if device is None else device
        self.self_loop = self_loop
        self.add_inverse_edge = add_inverse_edge
        
        # 本地 Rank，用于判断 dist_flag
        self.local_rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        
        self.stream = torch.cuda.Stream(device=device)

    def _get_dist_flag(self, nids: torch.Tensor) -> torch.Tensor:
        """
        判断节点是否为远程节点。
        返回 Bool Tensor: True = Remote (Stale), False = Local (Fresh).
        """

        return self.partition_state.get_part(nids) != self.local_rank

    def process(self, graph_blob: AsyncGraphBlob, start_t: int) -> List[StarryBatchData]:
        batch_res = []
        
        # 初始化 mapper (如果需要跨 Batch 维护，需放在类属性中)
        # 这里假设每个 Batch 都是独立的，mapper 仅用于还原 Global ID
        
        with torch.cuda.stream(self.stream):
            for i, g in enumerate(graph_blob.graph):
                if g is None: continue
                
                val_tuple = g.value()
                g_root = val_tuple.roots
                g_neg_roots = val_tuple.neg_roots
                g_neighbors = val_tuple.neighbors
                g_nid_mapper = val_tuple.nodes_remapper_id
                current_t = start_t + i
                root, neg_roots, mfgs, nid_mapper = AsyncBlock.to_block(
                    g_root, g_neg_roots, g_neighbors, g_nid_mapper, self_loop=self.self_loop, add_inverse_edge=self.add_inverse_edge,
                )
                
                dist_flags = []
                
                histories = []
                for layer_idx, b in enumerate(mfgs):
                    src_nids_local = b.srcdata['nid'] 
                    global_src_nids = nid_mapper[src_nids_local]
                    is_remote = self._get_dist_flag(global_src_nids)
                    dist_flags.append(is_remote)
                    
                    # Historical Embedding
                    if isinstance(self.history_state, list):
                        curr_hist_state = self.history_state[layer_idx] if layer_idx < len(self.history_state) else None
                    else:
                        curr_hist_state = self.history_state

                    if curr_hist_state is not None:
                        hist_emb, hist_ts = curr_hist_state.get(global_src_nids, current_t)
                        histories.append((hist_emb, hist_ts))
                    else:
                        histories.append(None)

                    if 'eid' in b.edata:
                        eids = b.edata['eid']
                        if 'f' in self.partition_data.edge_data: 
                            b.edata['f'] = self.partition_data.edge_data['f'].select(eids)

                    if layer_idx == 0:
                        node_feat = self.partition_data.node_data['f'].select(global_src_nids, current_t)
                        b.srcdata['x'] = node_feat

                unique_global_nids = nid_mapper
                if self.mailbox is not None:
                    mem_feat, mem_ts = histories[0]
                memory_list = [mem_feat, mem_ts]
                

                mail_feat, mail_ts = self.mailbox.get_message(unique_global_nids)
                mail_feat = mail_feat.reshape(unique_global_nids.size(0), -1)
                mailbox_list = [mail_feat, mail_ts]

                batch_res.append(StarryBatchData(
                    mfgs=mfgs,
                    dist_flag=dist_flags,
                    history=histories,
                    memory=memory_list,
                    mailbox=mailbox_list,
                    nid_mapper=nid_mapper,
                    roots=(root, neg_roots)
                ))
        
        event = torch.cuda.Event()
        event.record(self.stream)
        event.synchronize()
        
        return batch_res
    
class AsyncPipelineLoader:
    def __init__(self, 
                 dataset: StarryGLDataset,
                 graph_engine: pyGraph,
                 collator: GraphCollator,
                 sampling_config: dict,
                 partition_data: PartitionData, # 仅用于 prefetch
                 mode: str = 'CTDG',
                 prefetch_factor: int = 2,
                 negative_sampler = None):
        
        self.dataset = dataset
        self.graph = graph_engine
        self.collator = collator
        self.cfg = sampling_config
        self.partition_data = partition_data
        self.mode = mode
        
        self.prefetch_factor = prefetch_factor
        self.ready_queue = queue.Queue(maxsize=prefetch_factor)
        
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.task_iter = None
        self.negative_sampler = negative_sampler
        
    def __iter__(self):
        self._shutdown_worker()
        
        # 重置任务生成器
        print(self.cfg)
        self.task_iter = self.dataset.generate_tasks_smart(
            batch_size = self.cfg['batch_size'],
            sub_batch_size = self.cfg.get('sub_batch_size', 2000), # 默认每个子批次最多 2000 个事件
            max_span=(self.dataset.max_t.item() - self.dataset.min_t.item())//10,
            negative_sampler=self.negative_sampler,
           # shuffle_chunks=self.cfg.get('shuffle_chunks', True)
        )
        
        # 清空队列
        while not self.ready_queue.empty():
            try: self.ready_queue.get_nowait()
            except: pass
            
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        return self

    def __next__(self):
        batch = self.ready_queue.get()
        if batch is None:
            self._shutdown_worker()
            raise StopIteration
        return batch

    def _shutdown_worker(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.stop_event.set()
            # 消费残留以解锁 put
            while not self.ready_queue.empty():
                try: self.ready_queue.get_nowait()
                except: pass
            self.worker_thread.join(timeout=2.0)
        self.worker_thread = None

    def _submit_task(self, task):
        """
        第一阶段：提交 C++ 采样请求 & 触发 Cache Prefetch
        """
        # 计算时间步
        if self.mode == 'CTDG':
            time_steps = [task['time_start']]
            time_ends = [task['time_end']]
            start_t = 0 # Offset placeholder
        else: # DTDG
            time_steps = list(range(task['time_start'], task['time_end']))
            time_ends = [t + 1 for t in time_steps]
            start_t = task['time_start']

        # neg_samples = torch.empty(0)
        # if self.negative_sampler is not None:
        #     neg_samples = self.negative_sampler.sample(task['expected_size'])
        test_generate_kwargs = {
            'test_generate_samples':task.get('neg_roots', None),
            'test_generate_samples_ts': task.get('neg_roots_ts', None)
        }
        # 1. 构造异步图对象 (AsyncGraphBlob)
        # 这通常是非阻塞的，立即返回一个 Future/Promise 容器
        batch_graph = AsyncGraphBlob(
            self.graph,
            time_start=time_steps,
            time_end=time_ends,
            chunk_list=[task['chunks']],
            policy=self.cfg,
            test_generate_kwargs=test_generate_kwargs,
        )
        
        # 2. 触发 GPU Cache Prefetch (异步)
        # 通知 Cache 管理器将这些 chunk 的特征提前加载到 GPU
        if self.mode == 'CTDG':
            self.partition_data.prefetch_chunks(task['chunks'], 0)
        else:
            for t in range(task['time_start'], task['time_end']):
                self.partition_data.prefetch_chunks(task['chunks'], t)
        
        return (batch_graph, start_t)

    def _worker_loop(self):
        """
        Pipeline 核心循环:
        Fetch Task -> Submit Async Job -> [Wait previous Job] -> Collate -> Push to Queue
        """
        inflight_tasks = deque()
        
        try:
            # === Phase 1: 填满流水线 ===
            for _ in range(self.prefetch_factor):
                try:
                    task = next(self.task_iter)
                    item = self._submit_task(task)
                    inflight_tasks.append(item)
                except StopIteration:
                    break
            
            # === Phase 2: 稳态运行 ===
            while len(inflight_tasks) > 0 and not self.stop_event.is_set():
                
                # 1. 取出最早发出的请求
                graph_blob, start_t = inflight_tasks.popleft()
                
                # 2. 执行耗时的 Collate (等待采样完成 + 搬运特征)
                # 这会阻塞 Worker 线程，但此时 GPU/C++ 正在处理 inflight_tasks 中其他的任务
                final_batch = self.collator.process(graph_blob, start_t)
                
                # 3. 放入队列
                self.ready_queue.put(final_batch)
                
                # 4. 补充新任务，维持 Pipeline 深度
                try:
                    new_task = next(self.task_iter)
                    new_item = self._submit_task(new_task)
                    inflight_tasks.append(new_item)
                except StopIteration:
                    pass
            
            self.ready_queue.put(None) # End signal
            
        except Exception as e:
            print(f"DataLoader Worker Error: {e}")
            import traceback
            traceback.print_exc()
            self.ready_queue.put(None)