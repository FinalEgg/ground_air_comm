import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
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
                 max_power_dbm=30, area_size=1000.0, max_height=100.0, min_height=20.0):
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
        self.device = 'cpu' # 如果希望GPU加速计算，可以外部更改为 'cuda'
        self.params = UAVMIMOTensorParams(device=self.device)
        self.params.p_d = 10 ** (max_power_dbm / 10) # 将最大发射功率池从dBm转换为线性功率
        self.ch_model = BatchedMIMOChannel(self.M, self.K, self.N, params=self.params)
        
        # 基站通常处于固定的静态位置。此处将其固定在正方形网格结构中
        # 矩阵形状: (M, 3) -> (X, Y, Z)
        self.bs_pos = np.zeros((self.M, 3), dtype=np.float32)
        grid_width = int(np.sqrt(self.M))
        idx = 0
        for i in range(grid_width):
            for j in range(self.M // grid_width):
                if idx < self.M:
                    self.bs_pos[idx] = [
                        (i + 0.5) * (self.area_size / grid_width), 
                        (j + 0.5) * (self.area_size / (self.M // grid_width)), 
                        0.0 # 基站通常建在地面或低矮桅杆上，统一设其Z坐标为0
                    ]
                    idx += 1
                    
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

        # 内部状态：维护当前时间步所有的无人机坐标
        self.uav_pos = None

    def reset(self, seed=None, options=None):
        """
        环境重置操作：会重新随机初始化所有无人机(UAV)在定义区域内的位置。
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        # 随机设定无人机的三维空间分布
        x = np.random.uniform(0, self.area_size, self.K)
        y = np.random.uniform(0, self.area_size, self.K)
        z = np.random.uniform(self.min_h, self.max_h, self.K)
        self.uav_pos = np.stack([x, y, z], axis=-1).astype(np.float32) # (K, 3) 规模的坐标
        
        return self._get_obs(), {}

    def _get_obs(self):
        """
        获取当前环境的观测状态 (State)。
        包含了经过归一化处理后的无人机三维坐标以及其和所有基站通信的大尺度衰落特征 (beta_mk)。
        """
        # 我们需要在观测里加入网络容易学习的 beta_mk 特征，
        # 在这里执行快速的张量级计算获取信道数据。
        with torch.no_grad():
            t_bs = torch.tensor(self.bs_pos, dtype=torch.float32).unsqueeze(0) # (1, M, 3)
            t_uav = torch.tensor(self.uav_pos, dtype=torch.float32).unsqueeze(0) # (1, K, 3)
            
            D_mk, R_mk, theta_mk = self.ch_model.compute_distances_and_angles(t_bs, t_uav)
            beta_mk = self.ch_model.compute_large_scale_fading(D_mk, theta_mk) # (1, M, K)
        
        # 降维去除Batch维度
        beta_num = beta_mk.squeeze(0).numpy() # (M, K)
        
        # 融合网络需要的节点状态特征
        # uav_pos尺寸是 (K, 3), beta_num进行转置后的尺寸是 (K, M)
        features = np.concatenate([self.uav_pos, beta_num.T], axis=-1) # 合并得到 (K, 3 + M)
        
        # 观测空间归一化处理 (将实际坐标值缩放到 ~[0,1] 的范围更利于Actor网络拟合)
        features[:, 0:2] /= self.area_size
        features[:, 2] /= self.max_h
        
        # 大尺度衰落系数 beta 往往是一个极小的数值 (如 1e-9)，直接输入对神经网络非常不敏感。
        # 最好的方式是应用对数变换，以提取其相对大小幅度特征。
        features[:, 3:] = np.log10(features[:, 3:] + 1e-12) / 10.0 # 利用基于经验的除以10.0的缩放方法
        
        return features.flatten()

    def step(self, action):
        """
        环境交互步骤 (Step)。
        接收策略给出的动作并在内部应用动作约束。计算系统通信容量并生成奖励等信息。
        
        参数:
            action: 策略网络输出的原始动作值，应为长度等于 (M * K) 的一维平铺数组。(可能存在对数几率无边界的情形)
        返回:
            观测值 (next state)，当前步的奖励 (reward)，情境是否终止 (terminated)，是否阶段性截断 (truncated)，额外监控信息 (info)
        """
        # 1. 动作加工与整形阶段：利用 Softmax 将分配给多个无人机的动作约束和归一化
        action_matrix = action.reshape(self.M, self.K)
        
        # 为了保证基站可用总功率全分配(即基于无人机 K 这个轴进行求和始终为 1.0)
        # 用 NumPy 进行稳定的 Softmax 操作
        exp_a = np.exp(action_matrix - np.max(action_matrix, axis=1, keepdims=True))
        eta_mk = exp_a / np.sum(exp_a, axis=1, keepdims=True) # 归一化条件: sum_{k} (eta_mk) = 1
        
        # 2. 核心张量环境评估：利用定义好的信道模型完成物理规律模拟
        with torch.no_grad():
            t_bs = torch.tensor(self.bs_pos, dtype=torch.float32).unsqueeze(0)
            t_uav = torch.tensor(self.uav_pos, dtype=torch.float32).unsqueeze(0)
            t_eta = torch.tensor(eta_mk, dtype=torch.float32).unsqueeze(0)
            
            # 使用批处理维度完成空间拓扑关系的重建
            D_mk, R_mk, theta_mk = self.ch_model.compute_distances_and_angles(t_bs, t_uav)
            beta_mk = self.ch_model.compute_large_scale_fading(D_mk, theta_mk)
            gamma_mk = self.ch_model.compute_channel_estimation_variance(beta_mk)
            
            # 代入资源分配计划 (eta_mk) 获取到基于公式理论下限的香农定理吞吐量以及对应的接收到最终干噪比
            C_k, SINR = self.ch_model.compute_ergodic_capacities(beta_mk, gamma_mk, t_eta)
            
            c_k_np = C_k.squeeze(0).numpy() # 转为对用户 K 的 Numpy 数组 (K,)
        
        # 3. 强化学习的奖励 (Reward) 设计
        # A. 系统加权/单纯总和吞吐量 (Sum-rate)
        sum_rate = np.sum(c_k_np)
        
        # B. 建立业务质量 (QoS) 要求的约束以及相应的惩罚
        qos_penalty = 0.0
        if self.qos_method == 'static':
            # 当采用静态惩罚机制时：检测容量不满足规定目标的无人机，并施加大额度的乘子做静态惩罚
            violations = np.maximum(0, self.qos_threshold - c_k_np)
            qos_penalty = -50.0 * np.sum(violations) # 50 此处为经验推导设置的拉格朗日静态软惩罚超参数
        elif self.qos_method == 'lagrangian':
            # 当采用算法自适应基于拉格朗日约束强化学习时(如 PPO-Lagrangian 等安全RL):
            # 将具体不满足数值仅传递向算法层框架提供。本层环境不做直接的惩罚衰减。网络更新层面将使用外层拉格朗日代价计算器。
            violations = np.maximum(0, self.qos_threshold - c_k_np)
            qos_penalty = 0.0 # 因此此处设定的直接奖励不含惩罚
        
        reward = sum_rate + qos_penalty
        
        # 定义场景的物理边界 / 时间跨度 (Terminal state):
        # 当前架构设计成一个纯上下文赌博机 (Contextual Bandit) 即无后效型单步状态马尔可夫决策过程 (MDP)
        # 每决策一次即可算作完整执行过一次。随后即可告知框架 terminated=True
        # 注: 如果需要变成每 N 步或者序列衰减再返回的传统 RL 时，可另行改造这部分逻辑
        terminated = True
        truncated = False
        
        info = {
            "sum_rate": float(sum_rate),
            "capacities": c_k_np.tolist(),
            "violations_sum": float(np.sum(np.maximum(0, self.qos_threshold - c_k_np)))
        }
        
        return self._get_obs(), float(reward), terminated, truncated, info
