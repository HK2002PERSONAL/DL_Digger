import torch
import torch.nn as nn


class GATModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        # 选择 RNN 基模型：GRU 或 LSTM
        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,      # 输入特征维度
                hidden_size=hidden_size,# 隐藏层维度
                num_layers=num_layers,  # RNN 堆叠层数
                batch_first=True,       # 输入输出维度为 (batch, seq, feature)
                dropout=dropout,        # 层间 dropout（仅 num_layers>1 时生效）
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.hidden_size = hidden_size
        self.d_feat = d_feat

        # 注意力机制前的线性变换：将 RNN 输出映射到注意力空间
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)

        # 注意力权重参数 a，形状为 (hidden_size * 2, 1)
        # 用于计算拼接后的两个节点特征的注意力分数
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True

        # 全连接层：注意力聚合后的特征变换
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        # 输出层：将特征映射到预测值（回归任务输出 1 维）
        self.fc_out = nn.Linear(hidden_size, 1)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)  # 对第1维（行内）做 softmax，得到注意力权重

    def cal_attention(self, x, y):
        """
        计算图注意力权重（自注意力：x 和 y 通常是同一组节点）
        
        参数:
            x: (sample_num, dim)  -- 节点特征矩阵
            y: (sample_num, dim)  -- 节点特征矩阵（自注意力时与 x 相同）
        
        返回:
            att_weight: (sample_num, sample_num)  -- 注意力权重矩阵（每行和为1）
        """
        # x: (sample_num, hidden_size) -> (sample_num, hidden_size)
        x = self.transformation(x)
        # y: (sample_num, hidden_size) -> (sample_num, hidden_size)
        y = self.transformation(y)

        sample_num = x.shape[0]  # 节点数量，即 batch_size
        dim = x.shape[1]         # 特征维度，即 hidden_size

        # e_x: (sample_num, dim) -> expand -> (sample_num, sample_num, dim)
        # 效果：e_x[i, j, :] = x[i, :]  （第 i 个节点特征复制到第 i 行所有列）
        e_x = x.expand(sample_num, sample_num, dim)
        
        # e_y: (sample_num, sample_num, dim) -> transpose(0,1) -> (sample_num, sample_num, dim)
        # 效果：e_y[i, j, :] = x[j, :]  （第 j 个节点特征复制到第 j 列所有行，即转置后第 i 行）
        e_y = torch.transpose(e_x, 0, 1)

        # torch.cat((e_x, e_y), 2): (sample_num, sample_num, dim*2)
        # .view(-1, dim * 2): (sample_num * sample_num, dim * 2)
        # 效果：将所有节点对 (i,j) 的特征拼接，共 sample_num^2 对
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)

        # self.a: (hidden_size * 2, 1)
        # self.a_t: (1, hidden_size * 2)
        a_t = torch.t(self.a)

        # torch.t(attention_in): (dim * 2, sample_num^2)
        # self.a_t (1, dim*2) @ (dim*2, sample_num^2) = (1, sample_num^2)
        # .view(sample_num, sample_num): (sample_num, sample_num)
        # 效果：计算每对节点 (i,j) 的注意力原始分数
        attention_out = a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)

        # 应用 LeakyReLU 激活: (sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)

        # Softmax 归一化（按行）: (sample_num, sample_num)
        # 每行表示一个节点对所有节点（包括自己）的注意力分布，和为1
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: (batch_size, seq_len, d_feat)  -- 输入时间序列特征
        
        返回:
            output: (batch_size,)  -- 每个样本的预测值
        """
        # out: (batch_size, seq_len, hidden_size)
        # _: 最后一个时间步的隐状态（GRU/LSTM 的 hidden/cell，此处不使用）
        out, _ = self.rnn(x)

        # hidden: (batch_size, hidden_size)
        # 取 RNN 最后一个时间步的输出作为每个节点的特征表示
        hidden = out[:, -1, :]

        # att_weight: (batch_size, batch_size)
        # 自注意力：计算 batch 中每个样本（节点）对其他所有样本的注意力权重
        att_weight = self.cal_attention(hidden, hidden)

        # att_weight (batch_size, batch_size) @ hidden (batch_size, hidden_size)
        # = (batch_size, hidden_size)
        # + hidden (batch_size, hidden_size)  -- 残差连接（保留原始特征）
        hidden = att_weight.mm(hidden) + hidden

        # fc: (batch_size, hidden_size) -> (batch_size, hidden_size)
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)

        # fc_out: (batch_size, hidden_size) -> (batch_size, 1)
        # .squeeze(): (batch_size, 1) -> (batch_size,)
        return self.fc_out(hidden).squeeze()