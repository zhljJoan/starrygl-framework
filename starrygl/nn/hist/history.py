import torch
import torch.nn as nn

from starrygl.nn.model.time_encoder import TimeEncoder
class CompensationLayer(nn.Module):
    """
    延迟补偿层：融合历史嵌入和时间编码，以缓解远程节点的特征滞后问题。
    """
    def __init__(self, feature_dim: int, time_dim: int, hidden_dim: int = None):
        super().__init__()
        self.feature_dim = feature_dim
        out_dim = hidden_dim if hidden_dim is not None else feature_dim
        self.linear = nn.Linear(feature_dim + time_dim, out_dim)
        self.act = nn.ReLU()
        
    def forward(self, h_hist: torch.Tensor, dt: torch.Tensor, time_encoder: TimeEncoder):
        """
        Args:
            h_hist: 历史嵌入 [N, D]
            dt: 时间差 [N] (当前时间 - 历史嵌入记录时间)
            time_encoder: 时间编码模块
        """
        t_emb = time_encoder(dt.float()) # [N, time_dim]
        combined = torch.cat([h_hist, t_emb], dim=1)
        return self.act(self.linear(combined))
    
    
    import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedCompensationLayer(nn.Module):
    """
    基于神经流 (Neural Flow) 和门控机制的补偿层。
    模拟公式: h(t + dt) = h(t) + Gate * Velocity(h(t), dt) * dt
    """
    def __init__(self, feature_dim: int, time_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.velocity_net = nn.Sequential(
            nn.Linear(feature_dim + time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim) 
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid() 
        )
        
        if hidden_dim != feature_dim:
            self.out_proj = nn.Linear(feature_dim, hidden_dim)
        else:
            self.out_proj = nn.Identity()

    def forward(self, h_hist: torch.Tensor, dt: torch.Tensor, time_encoder: nn.Module):
        """
        Args:
            h_hist: 历史/旧的 Memory [N, D]
            dt: 时间差 [N]
            time_encoder: 时间编码器
        """
        if dt.dim() == 1: dt = dt.unsqueeze(1)
        
        t_emb = time_encoder(dt.float()) 
        if t_emb.dim() == 3 and t_emb.shape[1] == 1:
            t_emb = t_emb.squeeze(1)
            
        inp = torch.cat([h_hist, t_emb], dim=-1)
        velocity = self.velocity_net(inp)
        gate = self.gate_net(inp)
        h_comp = h_hist + gate * velocity
        return self.out_proj(h_comp)