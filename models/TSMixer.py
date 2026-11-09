# import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """
    基础 MLP 块：Linear -> ReLU -> Dropout -> Linear
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for fc in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.constant_(fc.bias, 0.0)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TemporalMixing(nn.Module):
    """
    时间维度混合：对每个特征独立地在时间维度上做 MLP 变换
    输入 [B, T, F] -> 转置为 [B, F, T] -> MLP -> 转置回 [B, T, F]
    """
    def __init__(self, seq_len, hidden_dim, dropout=0.3):
        super().__init__()
        self.mlp = MLPBlock(seq_len, hidden_dim, seq_len, dropout)
    
    def forward(self, x):
        # x: [B, T, F]
        x = x.permute(0, 2, 1)  # [B, F, T]
        x = self.mlp(x)
        x = x.permute(0, 2, 1)  # [B, T, F]
        return x


class FeatureMixing(nn.Module):
    """
    特征维度混合：对每个时间步独立地在特征维度上做 MLP 变换
    输入 [B, T, F] -> MLP on F dim -> [B, T, F]
    """
    def __init__(self, feat_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.mlp = MLPBlock(feat_dim, hidden_dim, feat_dim, dropout)
    
    def forward(self, x):
        # x: [B, T, F]
        x = self.mlp(x)
        return x


class TSMixerLayer(nn.Module):
    """
    单层 TSMixer：TemporalMixing + FeatureMixing，带残差连接和 BatchNorm
    """
    def __init__(self, seq_len, feat_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.temporal_mixing = TemporalMixing(seq_len, hidden_dim, dropout)
        self.feature_mixing = FeatureMixing(feat_dim, hidden_dim, dropout)
        self.norm1 = nn.BatchNorm1d(seq_len)
        self.norm2 = nn.BatchNorm1d(seq_len)
    
    def forward(self, x):
        # x: [B, T, F]
        # Temporal Mixing with residual
        residual = x
        x = self.temporal_mixing(x)
        x = x + residual
        x = self.norm1(x)  # BatchNorm on [B, T, F] -> normalize over T dimension
        
        # Feature Mixing with residual
        residual = x
        x = self.feature_mixing(x)
        x = x + residual
        x = self.norm2(x)
        
        return x


class TSMixer(nn.Module):
    """
    TSMixer: An All-MLP Architecture for Time Series Forecasting
    
    参数：
    - d_feat: 输入特征数 (默认 6)
    - seq_len: 时间序列长度
    - d_model: 隐藏层维度 (默认 64)
    - num_layers: Mixer 层数 (默认 2)
    - dropout: Dropout 率 (默认 0.3)
    
    输入：[batch, seq_len, d_feat]
    输出：[batch, 1] 预测值
    """
    def __init__(self, d_feat=6, seq_len=80, d_model=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.d_feat = d_feat
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(d_feat, d_model)
        
        # TSMixer 层
        self.layers = nn.ModuleList([
            TSMixerLayer(seq_len, d_model, d_model * 2, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层：取最后一个时间步
        self.output_projection = nn.Linear(d_model, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.constant_(self.input_projection.bias, 0.0)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, 0.0)
    
    def forward(self, x):
        # x: [B, T, F]
        x = self.input_projection(x)  # [B, T, d_model]
        
        for layer in self.layers:
            x = layer(x)
        
        # 取最后一个时间步
        x = x[:, -1, :]  # [B, d_model]
        x = self.output_projection(x)  # [B, 1]
        
        return x


class Feature_TSMixer(TSMixer):
    """
    特征提取版 TSMixer：
    - 继承 TSMixer，输出中间层特征（output_projection 之前的 d_model 维向量）
    - 用于 LightGBM 后训练流程
    
    输入：[batch, seq_len, d_feat]
    输出：[batch, d_model]
    """
    def forward(self, x):
        # x: [B, T, F]
        x = self.input_projection(x)  # [B, T, d_model]
        
        for layer in self.layers:
            x = layer(x)
        
        # 取最后一个时间步的特征（不经过 output_projection）
        x = x[:, -1, :]  # [B, d_model]
        
        return x


class MinDayTSMixer(nn.Module):
    """
    分钟频到日频的 TSMixer 封装（与 MinDayRNN 对应）
    用于处理分钟频数据，按 time_tag 采样日频特征
    
    参数：
    - d_feat: 输入特征数
    - lookback: 回看天数
    - time_tag: 每天的时间步数 (如 1d=1, five_minute=48, one_minute=240)
    - d_model: 隐藏层维度
    - num_layers: Mixer 层数
    - dropout: Dropout 率
    """
    def __init__(self, d_feat=6, lookback=20, time_tag=24, d_model=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lookback = lookback
        self.time_tag = time_tag
        self.seq_len = lookback * time_tag
        self.d_model = d_model
        
        self.tsmixer = TSMixer(
            d_feat=d_feat,
            seq_len=self.seq_len,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, x):
        # x: [B, lookback * time_tag, F]
        return self.tsmixer(x)