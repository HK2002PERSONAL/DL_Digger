import torch
from torch import nn

class BaseRNN(nn.Module):
    def __init__(self, d_feat=8, hidden_size=64, num_layers=1, dropout=0.3, rnn_type="GRU", bi_dir=False, ifbn=False, bias=True):
        super().__init__()
        self.hid_size = hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.bi_dir = bi_dir
        self.rnn_layer = num_layers
        self.ifbn = ifbn
        self._build_model(bias)
        self._initialize_weights()

    def _build_model(self, bias = True):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size, bias=bias))
        self.net.add_module("act", nn.Tanh())
        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            bidirectional=self.bi_dir,
            dropout=self.dropout if self.rnn_layer > 1 else 0,
            bias=bias
        )
        self.fc_out = nn.Linear(in_features=self.hid_size, out_features=1, bias=bias)
        if self.ifbn:
            self.bn = nn.BatchNorm1d(self.hid_size)

    def _initialize_weights(self):
        # 初始化 fc_in 和 fc_out 的权重
        for name, param in self.net.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)  # Xavier 初始化
            elif "bias" in name:
                nn.init.constant_(param, 0.0)  # 偏置初始化为 0
        
        # 初始化 RNN 层的权重
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:  # 输入到隐藏层的权重
                nn.init.xavier_uniform_(param)  # Xavier 初始化
            elif "weight_hh" in name:  # 隐藏到隐藏层的权重
                nn.init.orthogonal_(param)  # 正交初始化
            elif "bias" in name:  # 偏置
                nn.init.constant_(param, 0.0)
                # 如果是 LSTM，初始化遗忘门偏置为 1
                if "l0" in name and self.rnn_type.upper() == "LSTM":
                    hidden_size = param.size(0) // 4  # LSTM 的 bias 分为 4 部分
                    param.data[hidden_size:hidden_size * 2] = 1.0  # 遗忘门偏置初始化为 1

        # 初始化 fc_out 的权重
        for name, param in self.fc_out.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)  # Xavier 初始化
            elif "bias" in name:
                nn.init.constant_(param, 0.0)  # 偏置初始化为 0

        # 初始化 BatchNorm 层
        if self.ifbn:
            nn.init.constant_(self.bn.weight, 1.0)  # gamma 初始化为 1
            nn.init.constant_(self.bn.bias, 0.0)  # beta 初始化为 0
        
    def forward(self, inputs):
        """LSTM 前向传播。

        Args:
            inputs: [batch, seq_len, d_feat]
        """
        rnn_out, _ = self.rnn(self.net(inputs))  # [batch, seq_len, hidden_size]
        last_hidden = rnn_out[:, -1, :]
        if self.ifbn:
            last_hidden = self.bn(last_hidden)  # 应用 BatchNorm
        out = self.fc_out(last_hidden)
        return out
    
    
class DayDayRNN(BaseRNN):
    def __init__(self, d_feat=8, keep_seq=20, keep_tail=False, hidden_size=64, num_layers=1, dropout=0.3, rnn_type="GRU", bi_dir=False, ifbn=False):
        super().__init__(d_feat, hidden_size, num_layers, dropout, rnn_type, bi_dir, ifbn)
        self.keep_seq=keep_seq
        self.keep_tail = keep_tail

    def forward(self, inputs):
        rnn_out, _ = self.rnn(self.net(inputs)) # [batch, seq_len, hidden_size]

        if self.keep_tail == True:
            kept_hidden = rnn_out[:, :-self.keep_seq, :] # [batch, seqlen-keep_seq, hidden_size]
        else:
            kept_hidden = rnn_out[:, -self.keep_seq:, :] # [batch, keep_seq, hidden_size]
        
        if self.ifbn:
            kept_hidden = self.bn(kept_hidden)  # 应用 BatchNorm
            
        return kept_hidden
    
    
class MinDayRNN(BaseRNN):
    def __init__(self, d_feat=8, lookback=20, time_tag=16, hidden_size=64, num_layers=1, dropout=0.3, rnn_type="GRU", bi_dir=False, ifbn=False, bias = True):
        super().__init__(d_feat, hidden_size, num_layers, dropout, rnn_type, bi_dir, ifbn, bias)
        self.lookback = lookback
        self.time_tag = time_tag

    def forward(self, inputs:torch.Tensor):
        rnn_out, _ = self.rnn(self.net(inputs))  # [batch, seq_len, hidden_size]
        return rnn_out[:,self.time_tag-1::self.time_tag,:]
    

