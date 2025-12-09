from asyncio import Future, Queue
import collections
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from dgl import DGLGraph
import dgl

from starrygl.data import NodeState
from starrygl.data.collection import FastChunkCachedTensorData
from starrygl.graph.graph import AsyncBlock, AsyncGraphBlob, pyGraph
from starrygl.data.partitions import PartitionData



import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch import Tensor
from typing import *

import dgl
import dgl.function as fn

from dgl import DGLGraph
from dgl.heterograph import DGLBlock

import math
from torch_sparse import SparseTensor

from .partitions import PartitionData


import torch
import torch.distributed as dist
from typing import List, Dict, Optional, Iterator
import threading
import queue

# 假设这些是你之前定义的模块
# from .feature_cache import FastChunkCachedTensorData
# from .node_state import NodeState
# from .cpp_wrapper import pyGraph, starrygl_ops  # 你的C++封装

# =============================================================================
# 1. 数据集定义 (Task Generator)
# =============================================================================
import torch
import random
from typing import Iterator

class StarryGLDataset:
    def __init__(self, 
                 node_chunk_id: torch.Tensor, 
                 ts: torch.Tensor, 
                 rank: int = 0, 
                 world_size: int = 1):
        self.node_chunk_id = node_chunk_id
        self.ts = ts
        self.rank = rank
        self.world_size = world_size
        
        self.min_t = self.ts.min().item()
        self.max_t = self.ts.max().item()
        
        # 本地持有的所有 chunk ID
        self.num_chunks = int(node_chunk_id.max()) + 1
        self.local_chunks = [c for c in range(self.num_chunks) if c % world_size == rank]
        
    def generate_tasks_stream(
        self, 
        time_window: int, 
        chunks_per_batch: int = 16, # <--- 核心参数：每次只加载这么多 chunk
        shuffle_chunks: bool = True
    ) -> Iterator[dict]:
        """
        流式生成任务。
        策略：Time Window -> Shuffle Chunks -> Split into Mini-batches
        """
        curr = self.min_t
        step = 0
        
        while curr < self.max_t:
            end = min(curr + time_window, self.max_t)
            
            # 1. 获取当前时间段需要处理的 Chunks
            # (如果是全量训练，就是 self.local_chunks)
            # (如果是活跃 chunk 训练，可以在这里先筛选一遍 active chunks)
            candidate_chunks = self.local_chunks[:] 
            
            # 2. 随机打乱 Chunk 顺序 (保证随机性)
            if shuffle_chunks:
                random.shuffle(candidate_chunks)
            
            # 3. 将 Chunk 列表切分为多个 Mini-batches
            # 只有这样，Feature Cache 才能通过 LRU 机制循环利用显存
            num_batches = math.ceil(len(candidate_chunks) / chunks_per_batch)
            
            for i in range(num_batches):
                # 切片
                start_idx = i * chunks_per_batch
                end_idx = min((i + 1) * chunks_per_batch, len(candidate_chunks))
                batch_chunks = candidate_chunks[start_idx : end_idx]
                
                chunk_tensor = torch.tensor(batch_chunks, dtype=torch.long)
                
                yield {
                    'time_start': curr,
                    'time_end': end,
                    'chunks': chunk_tensor, # 这个 tensor 很小，不会导致 OOM
                    'step': step,
                    'sub_step': i # 标记是当前时间窗的第几个子批次
                }
            
            curr += time_window
            step += 1

class AsyncStarryGLBatch:
    def __init__(self,
                 tasks: List[Future],
                 ):
        self.tasks = tasks
        self.meta_graphs = []
        self.state = []
        
    def set(self, i, key, value):
        self.state[i][key] = value

class StarryGLDataLoader:
    def __init__(self, 
                 dataset: StarryGLDataset,
                 graph_engine: 'pyGraph', 
                 sampling_config: dict,
                 mode: str = 'CTDG',
                 prefetch_factor: int = 2,
                 partition_data: Optional[PartitionData] = None,
                 stale_node_state: Optional[NodeState] = None,
                 ): # 例如每次只训练 50% 的 Chunk
        
        self.dataset = dataset
        self.graph = graph_engine
        self.cfg = sampling_config
        self.mode = mode
        self.prefetch_factor = prefetch_factor
        self.partition_data = partition_data   
        self.stale_node_state = stale_node_state   
        self.task_iter = None
        self.task_in_flight = 0
        
        self.meta_queue = queue.Queue(maxsize=prefetch_factor)
        self.ready_queue = queue.Queue(maxsize=prefetch_factor)
        
        self.stop_event = threading.Event()
        self.threads = []
        
    def __iter__(self):
        
        self.stop_event.clear()
        self.task_iter = self.dataset.generate_tasks_stream(
            time_window=self.cfg['time_window'],
            chunks_per_batch=self.cfg['chunks_per_batch'],
            shuffle_chunks=self.cfg.get('shuffle_chunks', True)
        )
        
        self.task_in_flight = 0

        self.worker_thread = threading.Thread(
            target=self._prefetch_loop,
            args=(self.task_iter,),
            daemon=True
        )
        self.worker_thread.start()
        
        # 预热流水线
        for _ in range(self.prefetch_factor):
            if not self._submit_next():
                break
        
        return self

    def _submit_next(self):
        try:
            # 从生成器获取下一个任务 (Lazy Evaluation)
            task = next(self.task_iter)
        except StopIteration:
            return None
            
        # 1. 提交图采样 (C++ Async)
        # 这里的 chunk_list 已经是随机采样过的小集合了
        
        if self.mode == 'CTDG':
            batch_graph = AsyncGraphBlob(
                self.graph,
                time_start=[task['time_start']],
                time_end=[task['time_end']],
                chunk_list=[task['chunks']],
                policy=self.cfg,
                test_generate_kwargs={},
            )
            self.partition_data.prefetch_chunks(task['chunks'], 0)
            self.metadat_queue.append(batch_graph)
        else:
            batch_graph = AsyncGraphBlob(
                self.graph,
                time_start=[i for i in range(task['time_start'], task['time_end'])],
                time_end=[i+1 for i in range(task['time_start'], task['time_end'])],
                chunk_list=[task['chunks']],
                policy=self.cfg,
                test_generate_kwargs={},
            )
            self.metadat_queue.append(batch_graph)
            for i in range(task['time_start'], task['time_end']):
                # 预取每个时间段的 chunk 数据
                self.partition_data.prefetch_chunks(task['chunks'], i)
        
        self.task_in_flight += 1
        return batch_graph

    def __next__(self):
        # (保持原有逻辑不变)
        if self.task_in_flight == 0:
            raise StopIteration
            
        temp_res = self.metadat_queue.popleft()  # 弹出一个任务
        
        # 补充下一个任务
        if not self._submit_next():
            self.metadat_queue.append(None)
        self.task_in_flight -= 1
        
        if temp_res is None:
            raise StopIteration
            
        return temp_res
    
    def generate_snap_blob(self, graph: AsyncBlock, mapper: torch.Tensor, batch:AsyncStarryGLBatch, t: int = 0) -> AsyncStarryGLBatch:
        g = graph.value()
        root, neg_roots, mfgs, nid_mapper = AsyncBlock.to_block(
            g.root, g.neg_roots, g.neighbors, g.nid_mapper
        )
        for i,mfg in enumerate(mfgs):
            for l,b in enumerate(mfg): 
        node_feat = PartitionData.node_data['f'].select(
            self.partition_data, root, neg_roots, neighbors, nid_mapper
        )
        
    def assemble_batch(self, sampled_graph: AsyncGraphBlob) -> AsyncStarryGLBatch:
        graphs = self.meta_queue.popleft()
        for i, g in enumerate(graphs):
            if g is None:
                continue
            # 组装成 StarryGLBatch
            sampled_graph = self.generate_snap_blob(g, t)
            if sampled_graph is not None:
                return sampled_graph
        
                    

    


