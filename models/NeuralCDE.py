import torch
import torch.nn as nn

# ==================== 2. 门控向量场 (GRU-style) ====================
class GatedVectorField(nn.Module):
    """
    把 GRU 的门控机制注入 CDE 向量场。
    f(z) = u ⊙ MLP(r ⊙ z)，其中 r 是 reset gate，u 是 update gate。
    """
    def __init__(self, hidden, input_ch):
        super().__init__()
        self.hidden = hidden
        self.input_ch = input_ch
        
        # Reset gate: 控制哪些维度要"重置"后计算候选
        self.W_r = nn.Linear(hidden, hidden)
        self.b_r = nn.Parameter(torch.zeros(hidden))
        
        # Update gate: 控制最终向量场输出强度
        self.W_u = nn.Linear(hidden, hidden)
        self.b_u = nn.Parameter(torch.zeros(hidden))
        
        # 候选矩阵 MLP: 输入是 r ⊙ z，输出 reshape 为 (hidden, input_ch)
        self.candidate = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden * input_ch)
        )
        
    def forward(self, z):
        # z: (batch, hidden)
        
        # Reset gate
        r = torch.sigmoid(self.W_r(z) + self.b_r)  # (batch, hidden)
        
        # Update gate
        u = torch.sigmoid(self.W_u(z) + self.b_u)  # (batch, hidden)
        
        # 候选矩阵: 基于重置后的状态
        z_reset = r * z
        M = self.candidate(z_reset)  # (batch, hidden * input_ch)
        M = M.view(-1, self.hidden, self.input_ch)  # (batch, hidden, input_ch)
        
        # 门控: 逐 hidden channel 控制
        # u: (batch, hidden) -> (batch, hidden, 1) 广播到每个 input_ch
        u = u.unsqueeze(-1)  # (batch, hidden, 1)
        f_gated = u * M       # (batch, hidden, input_ch)
        
        return f_gated

# ==================== 3. 门控 Neural CDE ====================
class GatedNeuralCDE(nn.Module):

    # def __new__(cls, *args, **kwargs):
    #     # 1. 创建并正常初始化实例
    #     instance = super().__new__(cls)
    #     cls.__init__(instance, *args, **kwargs)
        
    #     # 2. 在内部进行编译并返回编译后的对象
    #     return torch.compile(instance)
        
    def __init__(self, input_ch, hidden: int = 64, out_dim: int = 1):
        super().__init__()
        self.hidden = hidden
        self.input_ch = input_ch
        
        # ζ_θ: 初始值网络
        self.initial = nn.Linear(input_ch, hidden)
        
        # 门控向量场
        self.vf = GatedVectorField(hidden, input_ch)
        
        self.readout = nn.Linear(hidden, out_dim)
        
    def forward(self, x):
        _, seq_len, _ = x.shape
        
        # 初始条件
        z = self.initial(x[:, 0])  # (batch, hidden)
        
        # 控制增量 ΔX (线性插值)
        delta_x = x[:, 1:] - x[:, :-1]  # (batch, seq_len-1, input_ch)
        
        # 门控 CDE 循环
        for i in range(seq_len - 1):
            f_mat = self.vf(z)  # (batch, hidden, input_ch)，已被门控调节
            
            dx = delta_x[:, i].unsqueeze(-1)       # (batch, input_ch, 1)
            dz = torch.bmm(f_mat, dx).squeeze(-1)   # (batch, hidden)
            
            z = z + dz  # Euler 步进
        
        return self.readout(z)
