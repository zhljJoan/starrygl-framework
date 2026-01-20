from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
import torch
from dgl.heterograph import DGLBlock

@dataclass
class StarryBatchData:
    """
    统一的 Batch 数据接口。
    训练循环(main.py) 和 模型(TGN) 将直接使用这个对象。
    """
    # 1. 图结构
    mfgs: List[DGLBlock]             
    history: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] # [Layer] list of (Embedding, Time)
    mailbox: Optional[List[torch.Tensor]] # [Feature, Time] 对应 Layer 0 的输入节点
    
    # 3. 辅助信息
    dist_flag: List[torch.Tensor]         # [Layer] Bool Mask, True 表示是远程节点(Halo)
    nid_mapper: torch.Tensor              # Layer 0 输入节点的 Global ID
    
    # 4. 任务目标
    roots: Tuple[torch.Tensor, torch.Tensor] # (Positive Roots, Negative Roots) 或者是 Node IDs
    
    # 5. 通信路由 (新架构核心)
    routes: List[Any] = None              # 每一层的 CommPlan，用于模型内同步