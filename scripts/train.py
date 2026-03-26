import os
import argparse
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
from tianshou.env import DummyVectorEnv
from tianshou.utils import TensorboardLogger
from tianshou.exploration import GaussianNoise
from tianshou.policy import DDPGPolicy, TD3Policy, SACPolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OffpolicyTrainer

# Local imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.uav_mimo_env import UavMimoEnv
from models.actor_critic import build_attention_actor_critic

def get_args():
    parser = argparse.ArgumentParser(description="Multi-UAV MIMO Power Allocation RL")
    
    # 算法与环境基础参数
    parser.add_argument('--algo', type=str, default='sac', choices=['ddpg', 'td3', 'sac'], help='RL Algorithm')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 通信拓扑参数
    parser.add_argument('--num-bs', type=int, default=4, help='Number of Base Stations (M)')
    parser.add_argument('--num-uav', type=int, default=10, help='Number of UAVs (K)')
    parser.add_argument('--num-antennas', type=int, default=16, help='Antennas per BS (N)')
    parser.add_argument('--qos-method', type=str, default='static', choices=['static', 'lagrangian'])
    parser.add_argument('--qos-threshold', type=float, default=1.5, help='Capacity threshold in bps/Hz')
    
    # 网络架构参数
    parser.add_argument('--hidden-dim', type=int, default=128, help='Attention hidden dimension')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--actor-hidden-sizes', type=int, nargs='*', default=[256, 128])
    parser.add_argument('--critic-hidden-sizes', type=int, nargs='*', default=[256, 128])
    
    # 训练超参数
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=5000)
    parser.add_argument('--step-per-collect', type=int, default=100)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--training-num', type=int, default=8, help='Number of train envs')
    parser.add_argument('--test-num', type=int, default=4, help='Number of test envs')
    
    # 强化学习特有参数
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate')
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=100000)
    
    return parser.parse_args()

def make_env(args):
    return UavMimoEnv(
        num_bs=args.num_bs, 
        num_uav=args.num_uav, 
        num_antennas=args.num_antennas,
        qos_method=args.qos_method,
        qos_threshold=args.qos_threshold
    )

def main():
    args = get_args()
    
    # ======= 设置随机种子 =======
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # ======= 1. 创建环境 =======
    # 由于底层全是张量/Numpy计算，无需多进程，DummyVectorEnv 可直接提供 Batch 能力
    train_envs = DummyVectorEnv([lambda: make_env(args) for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([lambda: make_env(args) for _ in range(args.test_num)])
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    
    # 提取维度信息
    temp_env = make_env(args)
    state_shape = temp_env.observation_space.shape
    action_shape = temp_env.action_space.shape
    max_action = temp_env.action_space.high[0]
    
    print(f"[{args.algo.upper()}] Setup -> State: {state_shape}, Action: {action_shape}, Device: {args.device}")

    # ======= 2. 构建 Agent 网络 =======
    is_sac = (args.algo == 'sac')
    actor, critic, critic2 = build_attention_actor_critic(args, state_shape, action_shape, is_sac=is_sac)
    
    # 设置优化器
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    if is_sac or args.algo == 'td3':
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # ======= 3. 实例化 Tianshou 策略 =======
    if args.algo == 'ddpg':
        policy = DDPGPolicy(
            actor=actor, actor_optim=actor_optim,
            critic=critic, critic_optim=critic_optim,
            tau=args.tau, gamma=args.gamma,
            exploration_noise=GaussianNoise(sigma=args.exploration_noise),
            estimation_step=1, action_space=temp_env.action_space
        )
    elif args.algo == 'td3':
        policy = TD3Policy(
            actor=actor, actor_optim=actor_optim,
            critic=critic, critic_optim=critic_optim,
            critic2=critic2, critic2_optim=critic2_optim,
            tau=args.tau, gamma=args.gamma,
            exploration_noise=GaussianNoise(sigma=args.exploration_noise),
            policy_noise=0.2, update_actor_freq=2, noise_clip=0.5,
            estimation_step=1, action_space=temp_env.action_space
        )
    elif args.algo == 'sac':
        # SAC 拥有自动温度调节 (alpha) 让网络决定探索力度
        # 在新版的 tianshou 中 SAC 参数名为 critic1, critic1_optim 而不是 critic
        policy = SACPolicy(
            actor=actor, actor_optim=actor_optim,
            critic1=critic, critic1_optim=critic_optim,
            critic2=critic2, critic2_optim=critic2_optim,
            tau=args.tau, gamma=args.gamma,
            alpha=0.2, # 可以替换为自适应 alpha (Tuple)
            estimation_step=1, action_space=temp_env.action_space
        )

    # ======= 4. 配置 Replay Buffer 和 Collector =======
    buffer = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    
    # 预先收集一些随机数据填充 Buffer
    train_collector.collect(n_step=args.batch_size * 2, random=True)

    # ======= 5. 配置日志环境 =======
    log_path = os.path.join('log', args.algo)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # ======= 6. 开启离线策略 (Off-policy) 训练循环 =======
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
        
    def stop_fn(mean_rewards):
        # 无人机总和容量达到预期即可提前停止，如果是严格的惩罚环境，reward至少得是正的
        return mean_rewards >= (args.num_uav * 4) # 假定平均每个UAV 4 bps/Hz 作为很高的预期

    print("开始训练...")
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=10, # 测试时跑几个 epoch(或者在这个Contextual Bandit里就是几个拓扑)
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        save_best_fn=save_best_fn,
        logger=logger,
        stop_fn=stop_fn
    ).run()
    
    print("\n========= 训练完成 =========")
    print(f"最终最佳平均奖励: {result.best_reward:.4f}")

if __name__ == '__main__':
    main()
