import torch
import torch.nn as nn

batch, seq_len, embed_dim, num_heads = 2, 10, 64, 8

mha = nn.MultiheadAttention(embed_dim, num_heads)
x = torch.randn(batch, seq_len, embed_dim)

# 你想加的 attention score bias，形状 (seq_len, seq_len)
# 这里用随机值示例，实际可以是相对位置编码、图网络的边权重等
score_bias = torch.randn(batch, batch) * 0.1

# # 如果同时需要因果掩码，把 -inf 填进 bias 矩阵里
# causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
# score_bias = score_bias.masked_fill(causal_mask, float('-inf'))

# 传入 attn_mask（float 类型）
out, weights = mha(x, x, x, attn_mask=score_bias)

print(weights)  # 可以看到 bias 对分布的影响