# -*- coding: utf-8 -*-
"""
FreTS-TSMixer (频域 -> 时域 串联版)
====================================
结构 (每个 Block 内部):
    1. FreqMixer:  沿时间维 rFFT -> 复数两层 MLP -> irFFT (先频域)
    2. TimeMixer:  时域 Linear(seq_len -> seq_len) (后时域)
    3. FeatureMixer: 特征维两层 MLP
    每个子层均为残差连接 + LayerNorm
整体: N 个 Block -> Temporal Projection -> 残差

注意: 模型不做输入归一化, 请在外部自行完成标准化 (如收益率化 / z-score)。

参考:
  - TSMixer (Chen et al., Google Research, KDD 2023)
  - FreTS   (Yi et al., NeurIPS 2023)

接口: 输入 (batch, seq_len, num_channels), 输出 (batch, pred_len, num_channels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 复数线性层 (FreTS)
# ---------------------------------------------------------------------------
class ComplexLinear(nn.Module):
    """复数线性层: y = x @ (Wr + i*Wi)"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.wr = nn.Parameter(torch.empty(in_features, out_features))
        self.wi = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.wr)
        nn.init.xavier_uniform_(self.wi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) 复数
        real = x.real @ self.wr - x.imag @ self.wi
        imag = x.real @ self.wi + x.imag @ self.wr
        return torch.complex(real, imag)


# ---------------------------------------------------------------------------
# 三个混合子层
# ---------------------------------------------------------------------------
class FreqMixer(nn.Module):
    """频域时间混合: rFFT(时间维) -> 复数两层 MLP -> irFFT, 残差 + LayerNorm"""

    def __init__(self, seq_len: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        n_freq = seq_len // 2 + 1
        self.lin1 = ComplexLinear(n_freq, n_freq)
        self.lin2 = ComplexLinear(n_freq, n_freq)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        x_t = x.transpose(1, 2)                            # (B, C, L)
        #   (B, C, F) 复数
        X = torch.fft.rfft(x_t, dim=-1)                    # pylint:disable=E1102

        Y = self.lin1(X)
        Y = torch.complex(F.relu(Y.real), F.relu(Y.imag))  # 复数 ReLU
        Y = torch.complex(self.dropout(Y.real), self.dropout(Y.imag))
        Y = self.lin2(Y)
        #  (B, C, L) 实数
        y = torch.fft.irfft(Y, n=self.seq_len, dim=-1)     # pylint:disable=E1102
        out = self.norm(y + x_t)                           # 残差 + LayerNorm
        return out.transpose(1, 2)                         # (B, L, C)


class TimeMixer(nn.Module):
    """时域时间混合: Linear(L -> L), 残差 + LayerNorm"""

    def __init__(self, seq_len: int, dropout: float):
        super().__init__()
        self.mlp = nn.Linear(seq_len, seq_len)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        x_t = x.transpose(1, 2)                            # (B, C, L)
        out = self.dropout(F.relu(self.mlp(x_t)))
        out = self.norm(out + x_t)
        return out.transpose(1, 2)                         # (B, L, C)


class FeatureMixer(nn.Module):
    """特征混合: 两层 MLP, 残差 + LayerNorm"""

    def __init__(self, num_channels: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(num_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return self.dropout(self.norm(out))


# ---------------------------------------------------------------------------
# 混合 Block: 频域 -> 时域 -> 特征
# ---------------------------------------------------------------------------
class MixerBlock(nn.Module):
    def __init__(self, seq_len: int, num_channels: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.freq_mixer = FreqMixer(seq_len, dropout)
        self.time_mixer = TimeMixer(seq_len, dropout)
        self.feature_mixer = FeatureMixer(num_channels, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.freq_mixer(x)
        x = self.feature_mixer(x)
        x = self.time_mixer(x)
        return x


# ---------------------------------------------------------------------------
# FreTS-TSMixer 主模型
# ---------------------------------------------------------------------------
class FreTSTSMixer(nn.Module):
    """
    FreTS-TSMixer 主模型
    
    Args:
        seq_len:      输入序列长度
        num_channels: 变量个数
        num_blocks:   混合 Block 层数
        hidden_dim:   特征混合隐藏维度
        dropout:      Dropout 比例
    
    输入: (batch, seq_len, num_channels)
    输出: (batch, 1)
    """

    def __init__(
        self,
        seq_len: int,
        num_channels: int,
        num_blocks: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim

        self.fc0 = nn.Linear(num_channels, hidden_dim)

        self.blocks = nn.ModuleList([
            MixerBlock(seq_len, hidden_dim, hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.decoder = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        out = self.fc0(x)
        for blk in self.blocks:
            out = blk(out)
        return self.decoder(out[:,-1,:])


class Feature_FreTSTSMixer(FreTSTSMixer):
    """
    特征提取版 FreTS-TSMixer：
    - 继承 FreTSTSMixer，输出中间层特征（decoder 之前的 hidden_dim 维向量）
    - 用于 LightGBM 后训练流程
    
    输入: (batch, seq_len, num_channels)
    输出: (batch, hidden_dim)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        out = self.fc0(x)
        for blk in self.blocks:
            out = blk(out)
        # 取最后一个时间步的特征（不经过 decoder）
        return out[:, -1, :]  # (B, hidden_dim)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cpu"

    seq_len, n_ch = 80, 6
    model = FreTSTSMixer(
        seq_len=seq_len,
        num_channels=n_ch,
        num_blocks=3,
        hidden_dim=64,
        dropout=0.1,
    ).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    data = torch.rand((2000, 80, 6))

    print(data.shape)

    print(model(data).shape)