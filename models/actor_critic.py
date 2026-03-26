from .attention_net import UAVAttentionNet
from tianshou.utils.net.continuous import Actor, ActorProb, Critic
import math
import numpy as np
import torch
import torch.nn as nn


class AttentionCriticPreprocess(nn.Module):
    def __init__(self, feature_net, obs_shape, action_shape, device='cpu'):
        super().__init__()
        self.feature_net = feature_net
        self.obs_dim = int(np.prod(obs_shape))
        self.action_dim = int(np.prod(action_shape))
        self.device = device
        self.output_dim = feature_net.output_dim + self.action_dim
        self.target_device = torch.device(device)

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, device=self.target_device, dtype=torch.float32)
        elif obs.device != self.target_device or obs.dtype != torch.float32:
            obs = obs.to(device=self.target_device, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        state_part = obs[:, :self.obs_dim]
        action_part = obs[:, self.obs_dim:self.obs_dim + self.action_dim]
        features, hidden = self.feature_net(state_part, state, info)
        return torch.cat([features, action_part], dim=1), hidden


def _init_linear_layer(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


def _init_mlp(mlp, final_gain):
    linear_layers = [module for module in mlp.model if isinstance(module, nn.Linear)]
    if not linear_layers:
        return

    for layer in linear_layers[:-1]:
        _init_linear_layer(layer, math.sqrt(2.0))

    _init_linear_layer(linear_layers[-1], final_gain)


def _initialize_actor(actor):
    _init_mlp(actor.mu, final_gain=1e-2)
    if getattr(actor, '_c_sigma', False):
        _init_mlp(actor.sigma, final_gain=1e-2)
    else:
        actor.sigma_param.data.fill_(-1.0)


def _initialize_critic(critic):
    _init_mlp(critic.last, final_gain=1e-2)

def build_attention_actor_critic(args, state_shape, action_shape, is_sac=False):
    """
    一个工厂函数，用于基于 Tianshou 官方的网络架构搭建搭载了 Attention 底座的 Actor/Critic。
    
    args: 包含网络隐藏层参数、设备信息等。
    state_shape: 原始展平后的状态维度 (K * (3+M))
    action_shape: 环境设定的动作维度 (M * K)
    is_sac: 是否为 SAC 构建网络(SAC 的 Actor 需要输出高斯分布的 mu 和 sigma)
    """
    device = getattr(args, 'device', 'cpu')
    
    # 构建共享的/独立的 Attention 特征提取器
    # 对于稳定训练，通常给 Actor 和 Critic 各自分配一个独立的 Attention 底座，避免梯度冲突
    actor_feature_net = UAVAttentionNet(
        num_uav=args.num_uav, 
        num_bs=args.num_bs, 
        hidden_dim=args.hidden_dim, 
        num_heads=args.num_heads, 
        device=device
    )
    
    critic_feature_net = UAVAttentionNet(
        num_uav=args.num_uav, 
        num_bs=args.num_bs, 
        hidden_dim=args.hidden_dim, 
        num_heads=args.num_heads, 
        device=device
    )
    
    if is_sac:
        actor = ActorProb(
            actor_feature_net,
            action_shape,
            hidden_sizes=args.actor_hidden_sizes,
            unbounded=True,
            device=device,
            preprocess_net_output_dim=actor_feature_net.output_dim,
        ).to(device)
    else:
        actor = Actor(
            actor_feature_net,
            action_shape,
            hidden_sizes=args.actor_hidden_sizes,
            max_action=1.0,
            device=device,
            preprocess_net_output_dim=actor_feature_net.output_dim,
        ).to(device)

    critic_preprocess = AttentionCriticPreprocess(
        critic_feature_net,
        state_shape,
        action_shape,
        device=device,
    )
    critic = Critic(
        critic_preprocess,
        hidden_sizes=args.critic_hidden_sizes,
        device=device,
        preprocess_net_output_dim=critic_preprocess.output_dim,
    ).to(device)

    critic2_feature_net = UAVAttentionNet(
        num_uav=args.num_uav, num_bs=args.num_bs,
        hidden_dim=args.hidden_dim, num_heads=args.num_heads,
        device=device
    )
    critic2_preprocess = AttentionCriticPreprocess(
        critic2_feature_net,
        state_shape,
        action_shape,
        device=device,
    )
    critic2 = Critic(
        critic2_preprocess,
        hidden_sizes=args.critic_hidden_sizes,
        device=device,
        preprocess_net_output_dim=critic2_preprocess.output_dim,
    ).to(device)

    if is_sac:
        _initialize_actor(actor)
    _initialize_critic(critic)
    _initialize_critic(critic2)

    return actor, critic, critic2
