import os
import argparse
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
# Phase 3: 切换为 Linux 下高性能的 ShmemVectorEnv，结合 Phase 1 的 GPU Tensor 环境实现极速并行
from tianshou.env import ShmemVectorEnv
from tianshou.utils import TensorboardLogger
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import OffpolicyTrainer

# Local imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.uav_mimo_env import UavMimoEnv
from models.actor_critic import build_attention_actor_critic

def get_args():
    parser = argparse.ArgumentParser(description="Multi-UAV MIMO Power Allocation RL (Linux Optimized)")
    
    # 算法与环境基础参数
    parser.add_argument('--algo', type=str, default='sac', choices=['ddpg', 'td3', 'sac'], help='RL Algorithm')
    # 默认设备改为 cuda，因为环境现在强制依赖 GPU Tensor
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 通信拓扑参数
    parser.add_argument('--num-bs', type=int, default=4, help='Number of Base Stations (M)')
    parser.add_argument('--num-uav', type=int, default=10, help='Number of UAVs (K)')
    parser.add_argument('--num-antennas', type=int, default=16, help='Antennas per BS (N)')
    parser.add_argument('--qos-method', type=str, default='static', choices=['static', 'lagrangian'])
    parser.add_argument('--qos-threshold', type=float, default=1.5, help='Capacity threshold in bps/Hz')
    
    # 网络架构参数
    # ...existing code...
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
    # ...existing code...
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate')
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=100000)
    
    return parser.parse_args()

def make_env_factory(args):
    # 工厂函数返回一个无参函数，该函数调用时创建环境
    # 重要: 将 device 传入环境，使其内部 Tensor 驻留 GPU
    def _init():
        return UavMimoEnv(
            num_bs=args.num_bs, 
            num_uav=args.num_uav, 
            num_antennas=args.num_antennas,
            qos_method=args.qos_method,
            qos_threshold=args.qos_threshold,
            device=args.device # Phase 1 新增参数
        )
    return _init

def main():
    args = get_args()
    
    # ======= 设置随机种子 =======
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # ======= 1. 创建环境 (Phase 3: ShmemVectorEnv 并行) =======
    # ShmemVectorEnv 在 Linux 上利用共享内存实现极低开销的跨进程通信
    # 注意: Windows 上通常只支持 DummyVectorEnv 或 SubprocVectorEnv，
    # ShmemVectorEnv 在 Windows 上可能会退化或报错。
    # 为生产环境 Linux 准备，我们强制使用 ShmemVectorEnv。
    # Side Note: 如果必须在 Windows 调试，临时改回 DummyVectorEnv。
    
    env_fns = [make_env_factory(args) for _ in range(args.training_num)]
    test_env_fns = [make_env_factory(args) for _ in range(args.test_num)]
    
    # 训练环境池
    train_envs = ShmemVectorEnv(env_fns)
    # 测试环境池
    test_envs = ShmemVectorEnv(test_env_fns)
    
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    
    # 提取维度信息
    # 临时创建一个虚拟环境实例来获取 space 信息
    dummy_env = make_env_factory(args)()
    state_shape = dummy_env.observation_space.shape
    action_shape = dummy_env.action_space.shape
    max_action = dummy_env.action_space.high[0]
    
    print(f"[{args.algo.upper()}] Setup -> State: {state_shape}, Action: {action_shape}, Device: {args.device}")
    print("Environment Mode: Native PyTorch Tensor on GPU + ShmemVectorEnv Multiprocessing")

    # ======= 2. 构建 Agent 网络 =======
    # ...existing code...
    is_sac = (args.algo == 'sac')
    actor, critic, critic2 = build_attention_actor_critic(args, state_shape, action_shape, is_sac=is_sac)
    
    # 设置优化器
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    
    # SAC 需要两个 Critic 优化器 (Critic1 和 Critic2)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # ======= 3. 实例化 Tianshou 策略 =======
    # 仅保留 SAC 作为示例，其他算法逻辑类似
    if args.algo == 'sac':
        policy = SACPolicy(
            actor=actor, actor_optim=actor_optim,
            critic1=critic, critic1_optim=critic_optim,
            critic2=critic2, critic2_optim=critic2_optim,
            tau=args.tau, gamma=args.gamma,
            alpha=0.2, 
            estimation_step=1, 
            action_space=dummy_env.action_space
        )
    else:
        raise NotImplementedError("This script currently focuses on SAC optimization.")

    # ======= 4. 配置 Replay Buffer 和 Collector =======
    # VectorReplayBuffer 会自动处理多环境的数据存储
    # 注意: 虽然环境返回的是 GPU Tensor，但在放入 Collector 前 Tianshou 还是会处理为 Batch 形式
    # 并且 ReplayBuffer 默认基于 Numpy/CPU 内存。
    # 理想情况下可以定制 Buffer 存 GPU，但标准 Tianshou 流程是:
    # Env(GPU) -> (maybe cpu copy by Shmem) -> Buffer(CPU) -> (batch sample) -> GPU Train
    # Phase 1 做的 Env GPU 化主要省去的是 step 内部的计算开销。
    buffer = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    
    # 预先收集
    train_collector.collect(n_step=args.batch_size * 5, random=True)

    # ======= 5. 配置日志环境 =======
    log_path = os.path.join('log', args.algo)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # ======= 6. 开启离线策略 (Off-policy) 训练循环 =======
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
        
    def stop_fn(mean_rewards):
        return mean_rewards >= (args.num_uav * 5) # 提高一点预期

    print("开始极速训练 (Linux Optimized)...")
    # ...existing code...
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=10, 
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
