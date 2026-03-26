import torch
import math

class UAVMIMOTensorParams:
    """
    保存信道模型的确定性物理参数。
    用于地对空(Ground-to-Air)通信系统的参数配置。
    """
    def __init__(self, device='cpu'):
        self.device = device
        # 视距(LoS)概率相关的环境参数 (根据具体环境设定，例如城市环境)
        self.xi_1 = 9.61
        self.xi_2 = 0.16
        
        # 路径损耗指数
        self.alpha1 = 2.0  # 视距 (LoS) 路径损耗指数
        self.alpha2 = 2.8  # 非视距 (NLoS) 路径损耗指数
        
        # 功率和噪声参数
        self.tau_p = 16.0       # 导频长度 (通常至少等于用户数K)
        self.p_u = 10**(10/10)  # 上行导频功率 (例如10dBm转换为线性值，基于噪声功率进行缩放)
        self.p_d = 10**(30/10)  # 下行总发射功率 (例如30dBm转换为线性值)
        self.noise_var = 1.0    # 归一化噪声功率方差 (假设信号功率直接等同于信噪比SNR)


class BatchedMIMOChannel:
    def __init__(self, num_bs, num_uav, num_antennas, params: UAVMIMOTensorParams = None):
        """
        基于PyTorch的计算图，用于计算地对空MIMO信道容量。
        针对批量(Batched)执行进行了优化 (B: 批次大小 / Batch size)。
        
        参数:
            num_bs: 基站(BS)的数量 (M)
            num_uav: 无人机(UAV)用户的数量 (K)
            num_antennas: 每个基站的天线数量 (N)
            params: 物理通道参数对象 (UAVMIMOTensorParams)
        """
        self.M = num_bs
        self.K = num_uav
        self.N = num_antennas
        self.params = params if params else UAVMIMOTensorParams()
        self.device = self.params.device

    def compute_distances_and_angles(self, bs_pos, uav_pos):
        """
        计算基站和无人机之间的3D距离、2D水平距离以及仰角。
        
        参数:
            bs_pos: 基站坐标张量，形状为 (B, M, 3) - (x, y, z)
            uav_pos: 无人机坐标张量，形状为 (B, K, 3) - (x, y, z)
            
        返回:
            D_mk: 3D空间距离，形状为 (B, M, K)
            R_mk: 2D水平距离，形状为 (B, M, K)
            theta_mk: 仰角 (单位：度)，形状为 (B, M, K)
        """
        # 扩展维度以便进行广播计算: bs_pos (B, M, 1, 3), uav_pos (B, 1, K, 3)
        bs_exp = bs_pos.unsqueeze(2)
        uav_exp = uav_pos.unsqueeze(1)
        
        # 差异向量: (B, M, K, 3)
        diff = bs_exp - uav_exp
        
        # 水平距离 R_mk (x和y方向的欧氏距离)
        R_mk = torch.norm(diff[..., :2], dim=-1)
        
        # 高度差 H (假设地面基站 z=0，或者使用精确的z轴高度差)
        H_mk = torch.abs(diff[..., 2])
        
        # 3D 空间距离 D_mk
        D_mk = torch.norm(diff, dim=-1)
        
        # 仰角 (度数)，对应公式(4)
        # 加上1e-9防止除以0
        theta_mk = (180.0 / math.pi) * torch.atan2(H_mk, R_mk + 1e-9)
        
        return D_mk, R_mk, theta_mk

    def compute_large_scale_fading(self, D_mk, theta_mk):
        """
        计算大尺度衰落系数。
        
        参数:
            D_mk: 3D距离矩阵，形状 (B, M, K)
            theta_mk: 仰角矩阵，形状 (B, M, K)
            
        返回:
            beta_mk: 大尺度衰落系数，形状 (B, M, K)
        """
        p = self.params
        
        # 视距 (LoS) 存在的概率，对应公式(3)
        # 使用 torch.exp 支持梯度追踪（如有需要），尽管给定位置时它基本是常数
        P_mk_L = 1.0 / (1.0 + p.xi_1 * torch.exp(-p.xi_2 * (theta_mk - p.xi_1)))
        P_mk_NL = 1.0 - P_mk_L
        
        # 期望的大尺度衰落，对应公式(5)
        # 加上一个很小的正数 epsilon (1e-9) 防止距离为0时除以0报错
        D_safe = D_mk + 1e-9
        beta_mk = P_mk_L * (D_safe ** (-p.alpha1)) + P_mk_NL * (D_safe ** (-p.alpha2))
        
        return beta_mk

    def compute_channel_estimation_variance(self, beta_mk):
        """
        计算信道估计的方差 gamma_mk。
        
        参数:
            beta_mk: 大尺度衰落系数，形状 (B, M, K)
            
        返回:
            gamma_mk: 信道估计方差，对应公式(9)，形状 (B, M, K)
        """
        p = self.params
        # 导频信噪比
        snr_pilot = p.tau_p * p.p_u
        gamma_mk = (snr_pilot * (beta_mk ** 2)) / (snr_pilot * beta_mk + 1.0)
        return gamma_mk

    def compute_ergodic_capacities(self, beta_mk, gamma_mk, eta_mk):
        """
        使用Massive MIMO闭式期望公式，计算各用户的遍历可达容量。
        此方法基于共轭波束成形(Conjugate Beamforming, CB)的期望分析边界，
        从而完全避免了复杂的矩阵实例化(如h_mk, g_mk)。
        
        参数:
            beta_mk: (B, M, K) - 真实的信道大尺度衰落
            gamma_mk: (B, M, K) - 估计的信道方差
            eta_mk: (B, M, K) - 功率分配系数，由RL Actor输出 (需满足归一化和约束)
        
        返回:
            C_k: (B, K) - 每个用户的吞吐量/容量 (bps/Hz)
            SINR: (B, K) - 信号干噪比
        """
        p = self.params
        N = self.N
        
        # eta_mk 是总发射功率的分配比例。
        # 动作空间受限于：对于每个基站 M，都有 sum_k(eta_mk) <= 1。
        
        # 对应公式(15) 期望信号 (Desired Signal, DS_k) 的期望值
        # 对于CB，E[g_{mk}^T \hat{g}_{mk}^*] = N * gamma_mk
        # DS_k = sqrt(p_d) * N * sum_{m} (eta_mk^(1/2) * gamma_mk)
        # 维度变化: (B, M, K) --在M维度求和--> (B, K)
        # 注意：不同的参考论文中，有时功率系数模块会直接使用eta而不开根号。
        # 这里按照用户的公式(15)，使用了 eta_mk^(1/2)。
        DS_k = math.sqrt(p.p_d) * N * torch.sum(torch.sqrt(eta_mk + 1e-9) * gamma_mk, dim=1)
        
        # 期望信号的功率项 (|DS_k|^2)
        signal_power = DS_k ** 2  # (B, K)
        
        # 对应公式(16) 波束成形不确定性 (Beamforming Uncertainty, BU_k) 方差
        # Var(BU_k) = p_d * sum_m (eta_mk * N * beta_mk * gamma_mk)
        BU_var = p.p_d * N * torch.sum(eta_mk * beta_mk * gamma_mk, dim=1) # (B, K)
        
        # 对应公式(17) 用户间干扰 (User Interference, UI_k_k') 方差
        # 用户 k' 对用户 k 造成的干扰
        # Var(UI_kk') = p_d * sum_m (eta_mk' * N * beta_mk * gamma_mk')
        # 我们需要计算所有 k' != k 的情况并对 k' 求和。
        # 作用在用户 k 上的总干扰：sum_{m} sum_{k'!=k} (p_d * eta_mk' * N * beta_mk * gamma_mk')
        # 首先计算包含所有 k' (包括 k'=k) 的无约束总和，然后再减去 k'=k 的本身干扰部分。
        
        # 在每个基站处，计算所有用户的功率分配与gamma乘积的总和
        # 形状: (B, M)
        sum_eta_gamma = torch.sum(eta_mk * gamma_mk, dim=2) 
        
        # 受到所有用户(包含自己)造成的总干扰
        # beta_mk: (B, M, K)
        # 将 sum_eta_gamma 扩展到 (B, M, 1) 以便与 beta_mk 相乘
        # total_interf_all = p_d * N * sum_m ( beta_mk * sum_k'(eta_mk' * gamma_mk') )
        total_interf_all = p.p_d * N * torch.sum(beta_mk * sum_eta_gamma.unsqueeze(2), dim=1) # (B, K)
        
        # 减去 k'=k 时产生的自我干扰部分 (它已在上一步被包含在 total_interf_all 中)
        # 自我干扰部分刚好等于 Var(BU_k)。因此 Interf(k'!=k) = total_interf_all - BU_var
        UI_var = total_interf_all - BU_var # (B, K)
        
        # 噪声功率 (已缩放为1，前文假设信号功率即为SNR)
        noise = p.noise_var
        
        # 对应公式(14) 信道容量 (香农公式)
        SINR = signal_power / (BU_var + UI_var + noise)
        C_k = torch.log2(1.0 + SINR) # (B, K)
        
        return C_k, SINR
