import os
import argparse
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
# Phase 3: 切换为 Linux 下高性能的 ShmemVectorEnv，结合 Phase 1 的 GPU Tensor 环境实现极速并行
from tianshou.env import DummyVectorEnv, ShmemVectorEnv
from tianshou.utils import TensorboardLogger
from tianshou.data import Batch, Collector, VectorReplayBuffer, to_numpy
from tianshou.policy import SACPolicy
from tianshou.trainer import OffpolicyTrainer

# Local imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.uav_mimo_env import UavMimoEnv
from models.actor_critic import build_attention_actor_critic
from models.simplex_policy import SimplexSACPolicy


def get_optimizer_lr(optimizer):
    return float(optimizer.param_groups[0]['lr'])


def make_eval_seeds(args):
    return [args.eval_seed_offset + idx for idx in range(args.diag_eval_episodes)]


def aggregate_metrics(metric_rows):
    summary = {}
    if not metric_rows:
        return summary

    metric_keys = metric_rows[0].keys()
    for key in metric_keys:
        values = np.asarray([row[key] for row in metric_rows], dtype=np.float64)
        summary[key] = float(values.mean())
        summary[f'{key}_std'] = float(values.std())
    return summary


def evaluate_equal_power_metrics(args, seeds):
    env = make_env_factory(args)()
    metric_rows = []
    action = np.full(env.action_space.shape, 1.0 / args.num_uav, dtype=np.float32)
    for seed in seeds:
        env.reset(seed=seed)
        _, _, _, _, info = env.step(action)
        metric_rows.append(info)
    return aggregate_metrics(metric_rows)


def evaluate_policy_metrics(args, policy, seeds):
    env = make_env_factory(args)()
    metric_rows = []
    was_training = policy.training
    policy.eval()
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            batch = Batch(obs=np.expand_dims(obs, axis=0), info=Batch())
            with torch.no_grad():
                result = policy(batch)
            action = policy.map_action(to_numpy(result.act))[0]
            _, _, _, _, info = env.step(action)
            metric_rows.append(info)
    finally:
        policy.train(was_training)
    return aggregate_metrics(metric_rows)


def log_metric_dict(writer, prefix, step, metrics):
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            writer.add_scalar(f'{prefix}/{key}', float(value), step)

def get_args():
    parser = argparse.ArgumentParser(description="Multi-UAV MIMO Power Allocation RL (Linux Optimized)")
    
    # 算法与环境基础参数
    parser.add_argument('--algo', type=str, default='sac', choices=['ddpg', 'td3', 'sac'], help='RL Algorithm')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--env-device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device used by environment tensors')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 通信拓扑参数
    parser.add_argument('--num-bs', type=int, default=4, help='Number of Base Stations (M)')
    parser.add_argument('--num-uav', type=int, default=10, help='Number of UAVs (K)')
    parser.add_argument('--num-antennas', type=int, default=16, help='Antennas per BS (N)')
    parser.add_argument('--qos-method', type=str, default='static', choices=['static', 'lagrangian'])
    parser.add_argument('--qos-threshold', type=float, default=1.5, help='Capacity threshold in bps/Hz')
    parser.add_argument('--include-qos-penalty', action='store_true', help='Include QoS penalty in the environment reward')
    parser.add_argument('--reward-scale', type=float, default=1e6, help='Linear scale applied to environment reward')
    parser.add_argument('--obs-beta-log10-center', type=float, default=-6.0, help='Center used to normalize log10(beta) features')
    parser.add_argument('--obs-beta-log10-scale', type=float, default=2.0, help='Scale used to normalize log10(beta) features')
    
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
    parser.add_argument('--update-per-step', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--training-num', type=int, default=8, help='Number of train envs')
    parser.add_argument('--test-num', type=int, default=4, help='Number of test envs')
    parser.add_argument('--episode-per-test', type=int, default=32, help='Episodes collected during each Tianshou evaluation round')
    parser.add_argument('--precollect-multiplier', type=int, default=20, help='Random warmup size as batch-size times this multiplier')
    parser.add_argument('--diag-eval-episodes', type=int, default=16, help='Fixed-seed episodes used for custom diagnostic evaluation')
    parser.add_argument('--eval-seed-offset', type=int, default=10000, help='Starting seed for fixed diagnostic evaluation episodes')
    parser.add_argument('--env-workers', type=int, default=0, help='Override vectorized env worker count, 0 uses training-num/test-num')
    parser.add_argument('--torch-num-threads', type=int, default=1, help='Torch intra-op CPU threads used in env processes')
    parser.add_argument('--torch-num-interop-threads', type=int, default=1, help='Torch inter-op CPU threads used in main process')
    parser.add_argument('--env-vector-type', type=str, default='auto', choices=['auto', 'shmem', 'subproc', 'dummy'], help='Vector env backend for CPU environments')
    parser.add_argument('--compile-model', action='store_true', help='Use torch.compile for actor and critics when available')
    
    # 强化学习特有参数
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--alpha-lr', type=float, default=1e-4, help='Learning rate for SAC temperature auto-tuning')
    parser.add_argument('--action-temperature', type=float, default=1.0, help='Softmax temperature used by simplex action projection')
    parser.add_argument('--simplex-eps', type=float, default=1e-8, help='Numerical floor used during simplex projection')
    parser.add_argument('--target-entropy-ratio', type=float, default=0.75, help='Target entropy ratio relative to maximum simplex entropy')
    # ...existing code...
    parser.add_argument('--gamma', type=float, default=0.0, help='Discount factor for single-step bandit training')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate')
    parser.add_argument('--fixed-alpha', type=float, default=0.2, help='SAC entropy coefficient used when auto-alpha is disabled')
    parser.add_argument('--target-entropy', type=float, default=None, help='Override target entropy for SAC auto-alpha')
    parser.add_argument('--reward-normalization', action=argparse.BooleanOptionalAction, default=False, help='Enable SAC reward normalization')
    parser.add_argument('--auto-alpha', action=argparse.BooleanOptionalAction, default=True, help='Enable SAC temperature auto-tuning')
    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--stop-reward', type=float, default=None, help='Optional early stop threshold on mean reward')
    
    return parser.parse_args()

def configure_torch_runtime(args):
    torch.set_num_threads(max(1, args.torch_num_threads))
    if hasattr(torch, 'set_num_interop_threads'):
        try:
            torch.set_num_interop_threads(max(1, args.torch_num_interop_threads))
        except RuntimeError:
            pass

def make_env_factory(args):
    # 工厂函数返回一个无参函数，该函数调用时创建环境
    def _init():
        torch.set_num_threads(max(1, args.torch_num_threads))
        return UavMimoEnv(
            num_bs=args.num_bs, 
            num_uav=args.num_uav, 
            num_antennas=args.num_antennas,
            qos_method=args.qos_method,
            qos_threshold=args.qos_threshold,
            device=args.env_device,
            include_qos_penalty=args.include_qos_penalty,
            reward_scale=args.reward_scale,
            obs_beta_log10_center=args.obs_beta_log10_center,
            obs_beta_log10_scale=args.obs_beta_log10_scale,
        )
    return _init

def resolve_vector_env_cls(args):
    if args.env_vector_type == 'dummy':
        return DummyVectorEnv
    if args.env_vector_type == 'subproc':
        return ts.env.SubprocVectorEnv
    if args.env_vector_type == 'shmem':
        return ShmemVectorEnv

    if args.env_device == 'cpu':
        return ShmemVectorEnv
    return DummyVectorEnv

def main():
    args = get_args()
    configure_torch_runtime(args)

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA training requested but no CUDA device is available.')
    if args.env_device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA env requested but no CUDA device is available.')

    if args.env_device == 'cuda' and args.training_num > 1:
        print('Warning: CUDA env with multiple workers falls back to DummyVectorEnv semantics. Consider env-device=cpu for throughput.')
    
    # ======= 设置随机种子 =======
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
    
    # ======= 1. 创建环境 =======
    env_worker_num = args.env_workers or args.training_num
    test_worker_num = min(args.test_num, env_worker_num) if args.env_workers else args.test_num
    env_fns = [make_env_factory(args) for _ in range(env_worker_num)]
    test_env_fns = [make_env_factory(args) for _ in range(test_worker_num)]
    env_cls = resolve_vector_env_cls(args)
    if args.env_device == 'cuda':
        env_cls = DummyVectorEnv

    train_envs = env_cls(env_fns)
    test_envs = env_cls(test_env_fns)
    
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    
    # 提取维度信息
    # 临时创建一个虚拟环境实例来获取 space 信息
    dummy_env = make_env_factory(args)()
    state_shape = dummy_env.observation_space.shape
    action_shape = dummy_env.action_space.shape
    
    print(f"[{args.algo.upper()}] Setup -> State: {state_shape}, Action: {action_shape}, Train Device: {args.device}, Env Device: {args.env_device}")
    print(f"Environment Mode: {env_cls.__name__} with {len(train_envs)} train envs / {len(test_envs)} test envs")

    # ======= 2. 构建 Agent 网络 =======
    is_sac = (args.algo == 'sac')
    actor, critic, critic2 = build_attention_actor_critic(args, state_shape, action_shape, is_sac=is_sac)
    if args.compile_model and hasattr(torch, 'compile'):
        actor = torch.compile(actor)
        critic = torch.compile(critic)
        critic2 = torch.compile(critic2)
    
    # 设置优化器
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    
    # SAC 需要两个 Critic 优化器 (Critic1 和 Critic2)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    target_entropy = args.target_entropy
    alpha = args.fixed_alpha
    alpha_optim = None
    if args.auto_alpha:
        if target_entropy is None:
            target_entropy = -float(args.target_entropy_ratio * args.num_bs * np.log(args.num_uav))
        log_alpha = torch.zeros(1, device=args.device, requires_grad=True)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)

    # ======= 3. 实例化 Tianshou 策略 =======
    # 仅保留 SAC 作为示例，其他算法逻辑类似
    if args.algo == 'sac':
        policy = SimplexSACPolicy(
            actor=actor, actor_optim=actor_optim,
            critic1=critic, critic1_optim=critic_optim,
            critic2=critic2, critic2_optim=critic2_optim,
            tau=args.tau, gamma=args.gamma,
            alpha=alpha,
            reward_normalization=args.reward_normalization,
            estimation_step=1, 
            action_space=dummy_env.action_space,
            num_bs=args.num_bs,
            num_uav=args.num_uav,
            action_temperature=args.action_temperature,
            simplex_eps=args.simplex_eps,
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
    warmup_steps = max(args.batch_size * args.precollect_multiplier, args.batch_size)
    train_collector.collect(n_step=warmup_steps, random=True)

    # ======= 5. 配置日志环境 =======
    log_path = os.path.join('log', args.algo)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    eval_seeds = make_eval_seeds(args)
    equal_power_metrics = evaluate_equal_power_metrics(args, eval_seeds)
    log_metric_dict(writer, 'baseline/equal_power', 0, equal_power_metrics)

    print(
        f"Baseline(Equal Power) -> reward={equal_power_metrics.get('reward', 0.0):.4f}, "
        f"sum_rate={equal_power_metrics.get('sum_rate', 0.0):.6f}, "
        f"min_rate={equal_power_metrics.get('min_rate', 0.0):.6f}"
    )
    print(
        f"Simplex Policy -> target_entropy={target_entropy if target_entropy is not None else args.fixed_alpha:.4f}, "
        f"action_temperature={args.action_temperature:.3f}"
    )

    # ======= 6. 开启离线策略 (Off-policy) 训练循环 =======
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def train_fn(epoch, env_step):
        writer.add_scalar('train/actor_lr', get_optimizer_lr(actor_optim), env_step)
        writer.add_scalar('train/critic_lr', get_optimizer_lr(critic_optim), env_step)
        if args.auto_alpha:
            writer.add_scalar('train/alpha', float(policy._alpha.item()), env_step)
            writer.add_scalar('train/alpha_lr', get_optimizer_lr(alpha_optim), env_step)
        policy.train()

    def test_fn(epoch, env_step):
        step = env_step if env_step is not None else epoch
        policy_metrics = evaluate_policy_metrics(args, policy, eval_seeds)
        log_metric_dict(writer, 'diagnostic/policy', step, policy_metrics)
        log_metric_dict(writer, 'diagnostic/equal_power', step, equal_power_metrics)

        for metric_name in ['reward', 'reward_raw', 'sum_rate', 'min_rate', 'mean_rate', 'max_rate', 'jain_fairness']:
            if metric_name in policy_metrics and metric_name in equal_power_metrics:
                gap = policy_metrics[metric_name] - equal_power_metrics[metric_name]
                writer.add_scalar(f'diagnostic/gap/{metric_name}', gap, step)

        if args.auto_alpha:
            writer.add_scalar('diagnostic/alpha', float(policy._alpha.item()), step)

    stop_fn = None
    if args.stop_reward is not None:
        def stop_fn(mean_rewards):
            return mean_rewards >= args.stop_reward

    print("开始训练...")
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.episode_per_test,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        stop_fn=stop_fn
    ).run()

    writer.flush()
    writer.close()
    
    print("\n========= 训练完成 =========")
    best_reward = result.get('best_reward') if isinstance(result, dict) else result.best_reward
    print(f"最终最佳平均奖励: {best_reward:.4f}")

if __name__ == '__main__':
    main()
