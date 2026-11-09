import torch
import torch.nn as nn
import torchcde

class GatedVectorField(nn.Module):
    """返回 (batch, hidden, input_ch)"""
    def __init__(self, hidden, input_ch):
        super().__init__()
        self.hidden = hidden
        self.input_ch = input_ch
        self.W_r = nn.Linear(hidden, hidden)
        self.b_r = nn.Parameter(torch.zeros(hidden))
        self.W_u = nn.Linear(hidden, hidden)
        self.b_u = nn.Parameter(torch.zeros(hidden))
        self.candidate = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden * input_ch)
        )
        
    def forward(self, z):
        r = torch.sigmoid(self.W_r(z) + self.b_r)
        u = torch.sigmoid(self.W_u(z) + self.b_u)
        z_reset = r * z
        M = self.candidate(z_reset)
        M = M.view(-1, self.hidden, self.input_ch)
        u = u.unsqueeze(-1)
        return u * M  # (batch, hidden, input_ch)

class VectorField(nn.Module):
    def __init__(self, vf):
        super().__init__()
        self.vf = vf
    
    def forward(self, t, z):
        return self.vf(z)  # (batch, hidden, input_ch)

class GatedNeuralCDE(nn.Module):
    def __init__(self, input_ch, hidden=64, out_dim=1):
        super().__init__()
        self.initial = nn.Linear(input_ch, hidden)
        self.vf = GatedVectorField(hidden, input_ch)
        self.readout = nn.Linear(hidden, out_dim)
    
    def forward(self, x):
        # 1. Hermite cubic spline with backward differences
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
        X = torchcde.CubicSpline(coeffs)
        
        # 2. 初始条件
        z0 = self.initial(x[:, 0])
        
        # 3. 求解 CDE
        z_T = torchcde.cdeint(
            X=X,
            func=VectorField(self.vf),
            z0=z0,
            t=X.grid_points,
            method='rk4',
            adjoint=True,
            rtol=1e-3,
            atol=1e-4
        )
        # z_T: (batch, seq_len, hidden)，取最后一步
        return self.readout(z_T[:, -1])