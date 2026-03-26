import torch
import torch.nn as nn
import numpy as np

class UAVAttentionNet(nn.Module):
    """
    自注意力网络 (Self-Attention Network) 作为 DRL 的特征提取底座 (preprocess_net)。
    负责自动解析 UAV 之间的空间分布与 M 根基站天线的信道大尺度衰落，提取全局干扰特征。
    """
    def __init__(self, num_uav, num_bs, hidden_dim=128, num_heads=4, device='cpu'):
        super().__init__()
        self.K = num_uav
        self.M = num_bs
        # 每一个 UAV 的特征维度为：三维坐标 (x, y, z) + M个基站感受到的大尺度衰落 beta_mk
        self.feature_dim = 3 + self.M 
        self.hidden_dim = hidden_dim
        self.device = device
        
        # 1. 词嵌入层 (Token Embedding): 将物理特征映射到高维空间
        self.param_embedding = nn.Linear(self.feature_dim, hidden_dim)
        
        # 2. 多头自注意力机制 (Multi-Head Self-Attention): 
        # - Q (Query): 某无人机对功率的渴求度和对干扰的敏感度
        # - K (Key): 其他无人机在空间上的干扰辐射轮廓
        # - V (Value): 综合特征的实际聚合
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True # 输入格式为 (Batch, Seq_len, Features)
        )
        
        # 3. 前馈神经网络与残差连接 (FFN & LayerNorm)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 返回给下游 Actor/Critic 的特征总维度 
        self.output_dim = self.K * hidden_dim
        
        self.to(device)

    def forward(self, obs, state=None, info={}):
        """
        obs 形如 (BatchSize, K * (3 + M)) 的一维展平 tensor 或 numpy 数组。
        由 Tianshou 的 Env 传来。
        """
        # 使用 as_tensor 避免重复内存拷贝，这是比手动 isinstance 判断更优的 torch 惯用写法
        # 如果 obs 已经在正确的 device 上，它几乎是零开销的
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            
        # 如果是单一样本，增加 Batch 维度
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
            
        B = obs.shape[0]
        
        # 步骤 1: 序列化 (Reshape into Token Sequence) 
        # (Batch, K, feature_dim)
        x = obs.view(B, self.K, self.feature_dim)
        
        # 步骤 2: 空间特征升维嵌入
        # (Batch, K, hidden_dim)
        x_emb = torch.relu(self.param_embedding(x))
        
        # 步骤 3: Attention 干扰解析
        # PyTorch 的 MultiheadAttention 会自动基于传入的 x_emb 计算内部的 Q, K, V
        attn_out, attn_weights = self.attention(x_emb, x_emb, x_emb)
        
        # 残差连接 (Residual Connection)
        x_add = self.norm1(x_emb + attn_out)
        
        # 前馈网络提纯
        ffn_out = torch.relu(self.fc1(x_add))
        x_out = self.norm2(x_add + ffn_out) # (Batch, K, hidden_dim)
        
        # 步骤 4: 展平回一维给 Tianshou Actor/Critic 全连接层使用
        out = x_out.view(B, -1) # (Batch, K * hidden_dim)
        
        return out, state

