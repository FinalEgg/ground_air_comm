import gymnasium as gym
from gymnasium import spaces
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
                 device='cpu'):
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
        
        # 内部张量计算引擎设置
        self.device = device # 外部指定设备
        self.params = UAVMIMOTensorParams(device=self.device)
        self.params.p_d = 10 ** (max_power_dbm / 10) # 将最大发射功率池从dBm转换为线性功率
        self.ch_model = BatchedMIMOChannel(self.M, self.K, self.N, params=self.params)
        
        # 基站通常处于固定的静态位置。此处将其固定在正方形网格结构中
        # 矩阵形状: (M, 3) -> (X, Y, Z)
        # 初始化阶段先用 Numpy 生成坐标，随后立即转为 GPU Tensor 驻留显存
        bs_pos_np = np.zeros((self.M, 3), dtype=np.float32)
        grid_width = int(np.sqrt(self.M))
        idx = 0
        for i in range(grid_width):
            for j in range(self.M // grid_width):
                if idx < self.M:
                    bs_pos_np[idx] = [
                        (i + 0.5) * (self.area_size / grid_width), 
                        (j + 0.5) * (self.area_size / (self.M // grid_width)), 
                        0.0 # 基站通常建在地面或低矮桅杆上，统一设其Z坐标为0
                    ]
                    idx += 1
        
        self.bs_pos = torch.tensor(bs_pos_np, device=self.device, dtype=torch.float32)
        
        # 状态空间 (Observation Space):
        # 维度包含：每架无人机的三维坐标 (3) + 基站数量 (M) 个大尺度衰落系数 beta_mk
        # 展平后为形状为 (K * (3 + M),) 的连续型一维向量状态
        uav_feat_dim = 3 + self.M
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.K * uav_feat_dim,), 
            dtype=np.float32
        )
        
        # 动作空间 (Action Space):
        # Tianshou中的Actor网络最典型的是输出在 [-1, 1] 范围的连续值。
        # 此环境期望接收一维大小为 M * K 的数组，此后会在 step 函数中被 Softmax 和截断以实施合法的功率按份分配。
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.M * self.K,), 
            dtype=np.float32
        )

        # 内部状态：维护当前时间步所有的无人机坐标 (Tensor)
        self.uav_pos = None

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
            
        # 随机设定无人机的三维空间分布 (直接在 GPU 上生成)
        # x, y: [0, area_size]
        # z: [min_h, max_h]
        # 形状: (K, 3)
        self.uav_pos = torch.rand((self.K, 3), device=self.device, dtype=torch.float32)
        # 缩放 x (dim=0) 和 y (dim=1)
        self.uav_pos[:, 0] = self.uav_pos[:, 0] * self.area_size
        self.uav_pos[:, 1] = self.uav_pos[:, 1] * self.area_size
        # 缩放 z (dim=2)
        self.uav_pos[:, 2] = self.uav_pos[:, 2] * (self.max_h - self.min_h) + self.min_h
        
        return self._get_obs(), {}

    def _get_obs(self):
        """
        获取当前环境的观测状态 (State)。
        包含了经过归一化处理后的无人机三维坐标以及其和所有基站通信的大尺度衰落特征 (beta_mk)。
        返回的是 Torch Tensor。
        """
        # 使用 PyTorch 上下文，无需梯度
        with torch.no_grad():
            # 扩展维度为 Batch=1 以适配 BatchedMIMOChannel: (1, M, 3) 和 (1, K, 3)
            # 这里的 self.bs_pos 和 self.uav_pos 已经是 Tensor 且在 device 上
            t_bs = self.bs_pos.unsqueeze(0)
            t_uav = self.uav_pos.unsqueeze(0)
            
            D_mk, R_mk, theta_mk = self.ch_model.compute_distances_and_angles(t_bs, t_uav)
            beta_mk = self.ch_model.compute_large_scale_fading(D_mk, theta_mk) # (1, M, K)
        
            # 移除 Batch 维度: (M, K)
            beta_tensor = beta_mk.squeeze(0)
            
            # --- 构建特征 ---
            # features 需要合并 uav_pos(K, 3) 和 beta 的转置(K, M)
            # beta_tensor.t() -> (K, M)
            # torch.cat 沿着 dim=1
            
            # 为了归一化，先拷贝一份 uav_pos (避免修改原始状态)
            norm_pos = self.uav_pos.clone()
            norm_pos[:, 0:2] /= self.area_size
            norm_pos[:, 2] /= self.max_h
            
            # 大尺度衰落系数对数变换
            # log10(beta + 1e-12) / 10.0
            norm_beta = torch.log10(beta_tensor.t() + 1e-12) / 10.0
            
            features = torch.cat([norm_pos, norm_beta], dim=1) # (K, 3+M)
            
            # 展平返回
            return features.flatten()

    def step(self, action):
        """
        环境交互步骤 (Step)。
        参数:
            action: 动作值。可能是 numpy array 也可能是 tensor。
        """
        # 1. 动作加工与整形阶段
        # 确保 action 是 Tensor 并位于正确设备
        if isinstance(action, np.ndarray):
            action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        elif isinstance(action, torch.Tensor):
             if action.device != torch.device(self.device):
                 action = action.to(self.device)
        
        # action shape: (M * K,)
        action_matrix = action.reshape(self.M, self.K)
        
        # 利用 PyTorch 原生 Softmax 操作
        # 对由 K 个用户构成的维度 (dim=1) 进行 Softmax，保证 sum_k(eta_mk) = 1
        # 之前的代码是: np.exp(x - max) / sum(exp)
        # F.softmax 会自动处理数值稳定性
        eta_mk = F.softmax(action_matrix, dim=1) # (M, K)
        
        # 2. 核心张量环境评估
        with torch.no_grad():
            t_bs = self.bs_pos.unsqueeze(0)
            t_uav = self.uav_pos.unsqueeze(0)
            t_eta = eta_mk.unsqueeze(0) # (1, M, K)
            
            # 物理信道计算
            D_mk, R_mk, theta_mk = self.ch_model.compute_distances_and_angles(t_bs, t_uav)
            beta_mk = self.ch_model.compute_large_scale_fading(D_mk, theta_mk)
            gamma_mk = self.ch_model.compute_channel_estimation_variance(beta_mk)
            
            # 计算容量
            C_k, SINR = self.ch_model.compute_ergodic_capacities(beta_mk, gamma_mk, t_eta)
            
            # C_k shape: (1, K) -> (K,)
            c_k_t = C_k.squeeze(0)
        
        # 3. 强化学习的奖励 (Reward) 设计
        # A. Sum-rate
        sum_rate = torch.sum(c_k_t)
        
        # B. QoS Penalty
        qos_penalty = torch.tensor(0.0, device=self.device)
        if self.qos_method == 'static':
            # violations = max(0, threshold - capacity)
            # torch.max(input, other)
            # constant tensor for threshold
            thresh_t = torch.tensor(self.qos_threshold, device=self.device)
            violations = torch.maximum(torch.tensor(0.0, device=self.device), thresh_t - c_k_t)
            qos_penalty = -50.0 * torch.sum(violations)
        
        reward = sum_rate + qos_penalty
        
        terminated = True
        truncated = False
        
        # Info 字典通常用于调试或日志，通常需要转为 CPU 数字/列表以便能够被 logger 正常记录
        # 如果追求极致速度，可精简或移除 info
        info = {
            "sum_rate": float(sum_rate.item()),
            # "capacities": c_k_t.cpu().tolist(), # 可选，耗时较多
            # "violations_sum": ...
        }
        
        # 注意: 
        # 为了适配常规 Gym 接口和 Tianshou 默认 Collector 行为，
        # 如果下游组件不支持 Tensor，这里可能需要 .cpu().numpy()。
        # 但既然我们的目标是全量 GPU 优化，我们这里直接返回 Tensor。
        # 若 Tianshou 在 CPU 上运行 Collector，它可能需要 Tensor -> Numpy 的转换，
        # 但这步开销无法避免，除非 Collector 也在 GPU 上。
        # 只要我们去掉了 step 内部的来回转换，就已经达成了 Phase 1 目标。
        
        return self._get_obs(), reward.item(), terminated, truncated, info
