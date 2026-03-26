import gymnasium as gym
from gymnasium import spaces
import math
import numpy as np
import torch
import torch.nn.functional as F
from .channel_models import BatchedMIMOChannel, UAVMIMOTensorParams

class UavMimoEnv(gym.Env):
    """
    地对空(Ground-to-Air) MIMO功率分配强化学习环境。
    本环境遵循 Gymnasium 接口规范，内部大量矩阵运算通过 PyTorch 实现加速。
    
    动作 (Action): 每个基站针对每一架无人机的功率分配系数 (会经过Softmax处理)。
    状态 (State): 所有无人机的三维空间坐标，以及每架无人机到各个基站的信道大尺度衰落系数。
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_bs=4, num_uav=10, num_antennas=16, 
                 qos_method='static', qos_threshold=1.5, 
                 max_power_dbm=30, area_size=1000.0, max_height=100.0, min_height=20.0,
                 device='cpu', include_qos_penalty=False, reward_scale=1e6,
                 obs_beta_log10_center=-6.0, obs_beta_log10_scale=2.0):
        super().__init__()
        
        # 物理网格约束
        self.area_size = area_size   # 活动区域的边长(米)，默认1000m
        self.max_h = max_height      # 无人机允许的最大高度，默认100m
        self.min_h = min_height      # 无人机允许的最小高度，默认20m
        
        # 通信节点配置
        self.M = num_bs              # 基站(BS)数量，默认4个
        self.K = num_uav             # 无人机(UAV)数量，默认10架
        self.N = num_antennas        # 每个基站的天线数量，默认16根
        
        # 强化学习相关的需求配置
        self.qos_method = qos_method         # 服务质量(QoS)保障的方法：'static'(静态惩罚) 或 'lagrangian'(拉格朗日乘数池化)
        self.qos_threshold = qos_threshold   # 目标QoS阈值 (代表要求的最低通信容量，单位bps/Hz)
        self.include_qos_penalty = include_qos_penalty
        self.reward_scale = reward_scale
        self.obs_beta_log10_center = obs_beta_log10_center
        self.obs_beta_log10_scale = max(obs_beta_log10_scale, 1e-6)
        
        # 内部张量计算引擎设置
        self.device = device # 外部指定设备
        self.torch_device = torch.device(device)
        self.params = UAVMIMOTensorParams(device=self.device)
        self.params.p_d = 10 ** (max_power_dbm / 10) # 将最大发射功率池从dBm转换为线性功率
        self.ch_model = BatchedMIMOChannel(self.M, self.K, self.N, params=self.params)
        
        # 基站通常处于固定的静态位置。此处将其固定在正方形网格结构中
        # 矩阵形状: (M, 3) -> (X, Y, Z)
        # 初始化阶段先用 Numpy 生成坐标，随后立即转为 GPU Tensor 驻留显存
        bs_pos_np = np.zeros((self.M, 3), dtype=np.float32)
        grid_rows = max(1, math.ceil(math.sqrt(self.M)))
        grid_cols = max(1, math.ceil(self.M / grid_rows))
        idx = 0
        for i in range(grid_rows):
            for j in range(grid_cols):
                if idx < self.M:
                    bs_pos_np[idx] = [
                        (i + 0.5) * (self.area_size / grid_rows), 
                        (j + 0.5) * (self.area_size / grid_cols), 
                        0.0 # 基站通常建在地面或低矮桅杆上，统一设其Z坐标为0
                    ]
                    idx += 1
        
        self.bs_pos = torch.tensor(bs_pos_np, device=self.torch_device, dtype=torch.float32)
        self.bs_pos_batch = self.bs_pos.unsqueeze(0)
        
        # 状态空间 (Observation Space):
        # 维度包含：每架无人机的三维坐标 (3) + 基站数量 (M) 个大尺度衰落系数 beta_mk
        # 展平后为形状为 (K * (3 + M),) 的连续型一维向量状态
        uav_feat_dim = 3 + self.M
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.K * uav_feat_dim,), 
            dtype=np.float32
        )
        
        # 动作空间 (Action Space):
        # Phase 3 起将动作显式建模为每个基站上的非负功率份额。
        # 环境会对每个基站对应的 K 维动作做非负截断并重新归一化到 simplex。
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.M * self.K,), 
            dtype=np.float32
        )

        # 内部状态：维护当前时间步所有的无人机坐标 (Tensor)
        self.uav_pos = None
        self.beta_cache = None
        self.gamma_cache = None
        self.obs_cache = None
        self.obs_cache_np = None
        self.zero_t = torch.tensor(0.0, device=self.torch_device, dtype=torch.float32)
        self.qos_threshold_t = torch.tensor(self.qos_threshold, device=self.torch_device, dtype=torch.float32)
        self.obs_eps = torch.tensor(self.params.numeric_eps, device=self.torch_device, dtype=torch.float32)

    def _project_action_to_simplex(self, action_matrix):
        action_matrix = torch.clamp_min(action_matrix, 0.0)
        action_sums = action_matrix.sum(dim=1, keepdim=True)
        zero_mask = action_sums <= self.obs_eps
        if torch.any(zero_mask):
            action_matrix = action_matrix.clone()
            action_matrix[zero_mask.expand_as(action_matrix)] = 1.0
            action_sums = action_matrix.sum(dim=1, keepdim=True)
        eta_mk = action_matrix / action_sums
        return eta_mk

    def _obs_to_numpy(self, obs):
        return obs.detach().cpu().numpy().astype(np.float32, copy=False)

    def _refresh_state_cache(self):
        with torch.inference_mode():
            t_uav = self.uav_pos.unsqueeze(0)
            D_mk, _, theta_mk = self.ch_model.compute_distances_and_angles(self.bs_pos_batch, t_uav)
            self.beta_cache = self.ch_model.compute_large_scale_fading(D_mk, theta_mk)
            self.gamma_cache = self.ch_model.compute_channel_estimation_variance(self.beta_cache)

            beta_tensor = self.beta_cache.squeeze(0)
            norm_pos = self.uav_pos.clone()
            norm_pos[:, 0:2] = (norm_pos[:, 0:2] / self.area_size) * 2.0 - 1.0
            norm_pos[:, 2] = ((norm_pos[:, 2] - self.min_h) / (self.max_h - self.min_h)) * 2.0 - 1.0
            beta_log10 = torch.log10(beta_tensor.t() + self.obs_eps)
            norm_beta = torch.clamp(
                (beta_log10 - self.obs_beta_log10_center) / self.obs_beta_log10_scale,
                min=-1.0,
                max=1.0,
            )
            self.obs_cache = torch.cat([norm_pos, norm_beta], dim=1).flatten()
            self.obs_cache_np = self._obs_to_numpy(self.obs_cache)

    def reset(self, seed=None, options=None):
        """
        环境重置操作：会重新随机初始化所有无人机(UAV)在定义区域内的位置。
        """
        # Gymnasium 的 super().reset(seed=seed) 会处理 seed 的相关逻辑
        super().reset(seed=seed)
        if seed is not None:
            # 即使主要逻辑在 GPU，也同步一下 numpy seed 以防万一
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.device.startswith('cuda'):
                torch.cuda.manual_seed_all(seed)
            
        # 随机设定无人机的三维空间分布 (直接在 GPU 上生成)
        # x, y: [0, area_size]
        # z: [min_h, max_h]
        # 形状: (K, 3)
        self.uav_pos = torch.rand((self.K, 3), device=self.torch_device, dtype=torch.float32)
        # 缩放 x (dim=0) 和 y (dim=1)
        self.uav_pos[:, 0] = self.uav_pos[:, 0] * self.area_size
        self.uav_pos[:, 1] = self.uav_pos[:, 1] * self.area_size
        # 缩放 z (dim=2)
        self.uav_pos[:, 2] = self.uav_pos[:, 2] * (self.max_h - self.min_h) + self.min_h
        self._refresh_state_cache()
        
        return self.obs_cache_np, {}

    def _get_obs(self):
        if self.obs_cache is None:
            self._refresh_state_cache()
        return self.obs_cache

    def step(self, action):
        """
        环境交互步骤 (Step)。
        参数:
            action: 动作值。可能是 numpy array 也可能是 tensor。
        """
        # 1. 动作加工与整形阶段
        # 确保 action 是 Tensor 并位于正确设备
        if isinstance(action, np.ndarray):
            action = torch.as_tensor(action, device=self.torch_device, dtype=torch.float32)
        elif isinstance(action, torch.Tensor):
             if action.device != self.torch_device or action.dtype != torch.float32:
                 action = action.to(device=self.torch_device, dtype=torch.float32)
        
        # action shape: (M * K,)
        action_matrix = action.reshape(self.M, self.K)
        
        eta_mk = self._project_action_to_simplex(action_matrix)
        
        # 2. 核心张量环境评估
        if self.beta_cache is None or self.gamma_cache is None:
            self._refresh_state_cache()

        with torch.inference_mode():
            t_eta = eta_mk.unsqueeze(0) # (1, M, K)
            # 计算容量
            C_k, sinr = self.ch_model.compute_ergodic_capacities(self.beta_cache, self.gamma_cache, t_eta)
            
            # C_k shape: (1, K) -> (K,)
            c_k_t = C_k.squeeze(0)
            sinr_t = sinr.squeeze(0)
        
        # 3. 强化学习的奖励 (Reward) 设计
        sum_rate = torch.sum(c_k_t)
        violations = torch.clamp_min(self.qos_threshold_t - c_k_t, 0.0)
        qos_penalty = self.zero_t.to(dtype=sum_rate.dtype)
        if self.include_qos_penalty and self.qos_method == 'static':
            qos_penalty = -50.0 * torch.sum(violations)

        reward_raw = sum_rate + qos_penalty
        reward = reward_raw * self.reward_scale
        
        terminated = True
        truncated = False
        
        # Info 字典通常用于调试或日志，通常需要转为 CPU 数字/列表以便能够被 logger 正常记录
        # 如果追求极致速度，可精简或移除 info
        sum_rate_sq = torch.square(sum_rate)
        sum_rate_denom = self.K * torch.sum(torch.square(c_k_t)) + c_k_t.new_tensor(self.params.numeric_eps)
        fairness = sum_rate_sq / sum_rate_denom
        action_entropy = -torch.sum(eta_mk * torch.log(eta_mk + self.params.numeric_eps), dim=1).mean()

        info = {
            "reward": float(reward.item()),
            "reward_raw": float(reward_raw.item()),
            "reward_scale": float(self.reward_scale),
            "sum_rate": float(sum_rate.item()),
            "min_rate": float(torch.min(c_k_t).item()),
            "max_rate": float(torch.max(c_k_t).item()),
            "mean_rate": float(torch.mean(c_k_t).item()),
            "mean_sinr": float(torch.mean(sinr_t).item()),
            "max_sinr": float(torch.max(sinr_t).item()),
            "jain_fairness": float(fairness.item()),
            "qos_threshold": float(self.qos_threshold),
            "qos_violation_count": int(torch.count_nonzero(violations > 0.0).item()),
            "qos_violation_gap": float(torch.sum(violations).item()),
            "action_entropy": float(action_entropy.item()),
            "action_max_share": float(torch.max(eta_mk).item()),
            "action_min_share": float(torch.min(eta_mk).item()),
        }
        
        # 注意: 
        # 为了适配常规 Gym 接口和 Tianshou 默认 Collector 行为，
        # 如果下游组件不支持 Tensor，这里可能需要 .cpu().numpy()。
        # 但既然我们的目标是全量 GPU 优化，我们这里直接返回 Tensor。
        # 若 Tianshou 在 CPU 上运行 Collector，它可能需要 Tensor -> Numpy 的转换，
        # 但这步开销无法避免，除非 Collector 也在 GPU 上。
        # 只要我们去掉了 step 内部的来回转换，就已经达成了 Phase 1 目标。
        
        return self.obs_cache_np, float(reward.item()), terminated, truncated, info
