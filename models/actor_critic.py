from .attention_net import UAVAttentionNet
from tianshou.utils.net.continuous import Actor, ActorProb, Critic
from tianshou.utils.net.common import Net
import torch.nn as nn

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
    
    # 特征提取后的输出维度是 K * hidden_dim
    dense_input_dim = actor_feature_net.output_dim
    
    if is_sac:
        actor_net = Net(state_shape, hidden_sizes=args.actor_hidden_sizes, device=device)
        actor = ActorProb(
            actor_net, action_shape, unbounded=True,
            device=device
        ).to(device)
        actor.preprocess = actor_feature_net
    else:
        actor_net = Net(state_shape, hidden_sizes=args.actor_hidden_sizes, device=device)
        actor = Actor(
            actor_net, action_shape, max_action=1.0,
            device=device
        ).to(device)
        actor.preprocess = actor_feature_net

    critic_net = Net(
        state_shape, action_shape=action_shape, hidden_sizes=args.critic_hidden_sizes,
        concat=True, device=device
    )
    critic = Critic(
        critic_net, device=device
    ).to(device)
    critic.preprocess = critic_feature_net

    critic2_feature_net = UAVAttentionNet(
        num_uav=args.num_uav, num_bs=args.num_bs,
        hidden_dim=args.hidden_dim, num_heads=args.num_heads,
        device=device
    )
    critic2_net = Net(
        state_shape, action_shape=action_shape, hidden_sizes=args.critic_hidden_sizes,
        concat=True, device=device
    )
    critic2 = Critic(
        critic2_net, device=device
    ).to(device)
    critic2.preprocess = critic2_feature_net

    return actor, critic, critic2
