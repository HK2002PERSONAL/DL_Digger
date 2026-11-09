# import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

class Chomp1d(nn.Module):
    """
    用于裁剪掉因因果卷积而在右侧填充（Padding）的多余时序长度
    保证输入和输出的时序长度 T 一致
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    TCN 的核心残差块：包含两层空洞因果卷积 + 激活 + Dropout + 残差连接
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # 第一层因果空洞卷积
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二层因果空洞卷积
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # 如果输入通道和输出通道不一致，用 1x1 卷积进行下采样对齐
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """
    接收 (B, T, F) 输入，输出 (B,) 目标值的 TCN 模型
    """
    def __init__(self, input_size, num_channels=None, kernel_size=3, dropout=0.2):
        """
        Args:
            input_size: 输入特征数 (F)
            num_channels: 每一层 TCN 的输出通道数列表，例如 [32, 64, 128]
            kernel_size: 卷积核大小
        """
        super(TCNModel, self).__init__()
        layers = []
        if num_channels is None:
            num_channels = [64, 128, 128, 64]
        num_levels = len(num_channels)
        
        # 逐层构建 TemporalBlock，膨胀系数 (dilation) 随层数指数级增长
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # 为了保证因果性，padding 大小必须为 (kernel_size - 1) * dilation
            padding = (kernel_size - 1) * dilation_size
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        
        # 线性层：将最后一层 TCN 提取的特征映射到 1 个输出维度
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        """
        x 形状: (B, T, F)
        """
        # 1. PyTorch 的 Conv1d 期待形状为 (B, F, T)，所以需要转换轴
        x = x.transpose(1, 2) 
        
        # 2. 通过 TCN 网络，输出形状: (B, num_channels[-1], T)
        output = self.network(x)
        
        # 3. 提取时序维度的最后一个时间步 (Last Token) 作为全局特征
        # 形状变为: (B, num_channels[-1])
        last_step_feat = output[:, :, -1]
        
        # 4. 通过线性层映射到 (B, 1)
        out = self.linear(last_step_feat)
        
        # 5. 压平最后的维度，将 (B, 1) 变成 (B,)
        return out.squeeze(-1)
