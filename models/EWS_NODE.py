import torch
import torch.nn as nn
from typing import Tuple

# ==========================================
# 1. JIT 编译加速的 3阶 EWS 核心计算图
# ==========================================
@torch.jit.script
def ews_core_forward(dX: torch.Tensor, A_exp: torch.Tensor, depth: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    使用 TorchScript 静态编译的核心迭代循环
    完全消除 Python 循环开销，并对张量操作进行底层优化
    """
    batch_size = dX.size(0)
    T_minus_1 = dX.size(1)
    F_dim = dX.size(2)
    
    # 显式初始化各阶状态
    s1 = torch.zeros((batch_size, F_dim), device=dX.device, dtype=dX.dtype)
    s2 = torch.zeros((batch_size, F_dim, F_dim), device=dX.device, dtype=dX.dtype)
    s3 = torch.zeros((batch_size, F_dim, F_dim, F_dim), device=dX.device, dtype=dX.dtype)
    
    for t in range(T_minus_1):
        dx_t = dX[:, t, :]
        
        # 3阶更新
        if depth >= 3:
            temp1 = torch.einsum('im,bmno->bino', A_exp, s3)
            temp2 = torch.einsum('jn,bino->bijo', A_exp, temp1)
            s3_faded = torch.einsum('ko,bijo->bijk', A_exp, temp2)
            tensor_prod3 = torch.einsum('bij,bk->bijk', s2, dx_t)
            s3 = s3_faded + tensor_prod3
            
        # 2阶更新
        if depth >= 2:
            temp = torch.einsum('im,bmn->bin', A_exp, s2)
            s2_faded = torch.einsum('jn,bin->bij', A_exp, temp)
            tensor_prod2 = torch.einsum('bi,bj->bij', s1, dx_t)
            s2 = s2_faded + tensor_prod2
            
        # 1阶更新
        s1_faded = torch.einsum('ij,bj->bi', A_exp, s1)
        s1 = s1_faded + dx_t
        
    return s1, s2, s3


# ==========================================
# 2. 带有 B 算子开关的 EWSLayer
# ==========================================
class EWSLayer(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 4, depth: int = 3, use_projection: bool = False):
        """
        :param input_dim: 原始输入特征数 F
        :param latent_dim: B 算子投影到的隐空间维度（仅在 use_projection=True 时有效）
        :param depth: 签名阶数 (1, 2, 3)
        :param use_projection: 是否启用可学习参数 B (线性投影矩阵)，默认关闭 (False)
        """
        super().__init__()
        self.depth = depth
        self.use_projection = use_projection
        
        # 静态确定在 CDE 迭代时使用的核心特征维度
        if self.use_projection:
            self.core_dim = latent_dim
            # 实例化可学习参数 B
            self.B = nn.Linear(input_dim, latent_dim, bias=False)
        else:
            self.core_dim = input_dim
            # 即使不用 B，为了让 JIT 编译器在类型检查时不报错，我们占位一个 nn.Identity
            self.B = nn.Identity()
        
        # 跨通道耦合算子 A 作用于 core_dim
        self.A = nn.Parameter(torch.randn(self.core_dim, self.core_dim) * 0.02)
        
        # 动态计算最终输出给 NODE 的特征维度 K
        self.output_dim = sum([self.core_dim ** d for d in range(1, depth + 1)])

    def forward(self, x):
        # 1. 根据开关状态，选择是否通过 B 算子映射
        # JIT 编译器在编译时会根据 use_projection 的常数布尔值直接优化掉无效分支
        if self.use_projection:
            x_core = self.B(x)
        else:
            x_core = x
            
        # 2. 计算特征路径增量
        dX = x_core[:, 1:, :] - x_core[:, :-1, :]
        A_exp = torch.matrix_exp(self.A)
        
        # 3. 调用 JIT 编译的核心加速函数
        s1, s2, s3 = ews_core_forward(dX, A_exp, self.depth)
        
        # 4. 特征拼接展平输出
        batch_size = x.size(0)
        if self.depth == 1:
            return s1
        elif self.depth == 2:
            return torch.cat([s1.view(batch_size, -1), s2.view(batch_size, -1)], dim=-1)
        else:
            return torch.cat([
                s1.view(batch_size, -1), 
                s2.view(batch_size, -1), 
                s3.view(batch_size, -1)
            ], dim=-1)


# ==========================================
# 3. NODE 核心树结构模块 (保持高效的并行 Sparsemax)
# ==========================================
class DenseODT(nn.Module):
    def __init__(self, input_dim, num_trees, depth, output_dim=1):
        super().__init__()
        self.num_trees = num_trees
        self.depth = depth
        self.output_dim = output_dim
        
        self.feature_weights = nn.Parameter(torch.randn(num_trees, depth, input_dim) * 0.1)
        self.thresholds = nn.Parameter(torch.randn(num_trees, depth) * 0.1)
        self.scalings = nn.Parameter(torch.ones(num_trees, depth))
        self.leaves = nn.Parameter(torch.randn(num_trees, 2**depth, output_dim) * 0.1)

    def sparsemax(self, X, dim: int = -1):
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val
        sort_X, _ = torch.sort(X, dim=dim, descending=True)
        shape = list(X.shape)
        rank = torch.arange(1, shape[dim] + 1, device=X.device, dtype=X.dtype)
        view = [1] * len(shape)
        view[dim] = -1
        rank = rank.view(view)
        cumsum_X = torch.cumsum(sort_X, dim=dim)
        cond = 1 + rank * sort_X > cumsum_X
        idx = torch.clamp(torch.sum(cond, dim=dim, keepdim=True) - 1, min=0)
        gather_cumsum = torch.gather(cumsum_X, dim=dim, index=idx)
        gather_rank = torch.gather(rank.expand_as(cumsum_X), dim=dim, index=idx)
        tau = (gather_cumsum - 1) / gather_rank
        return torch.clamp(X - tau, min=0)

    def forward(self, x):
        batch_size = x.shape[0]
        flat_weights = self.feature_weights.view(-1, x.shape[-1])
        sparse_flat_weights = self.sparsemax(flat_weights)
        weights = sparse_flat_weights.view(self.num_trees, self.depth, x.shape[-1])
        
        context = torch.einsum('bk,ntk->bnt', x, weights)
        t = (context - self.thresholds) * self.scalings
        
        p_right = torch.clamp((t + 1) / 2, min=0, max=1)
        p_left = 1.0 - p_right
        
        leaf_probs = torch.ones(batch_size, self.num_trees, 1, device=x.device, dtype=x.dtype)
        for d in range(self.depth):
            pl = p_left[:, :, d:d+1]
            pr = p_right[:, :, d:d+1]
            leaf_probs = torch.cat([leaf_probs * pl, leaf_probs * pr], dim=-1)
            
        out = torch.einsum('bnl,nlo->bo', leaf_probs, self.leaves)
        return out


# ==========================================
# 4. 完整的联合可编译模型
# ==========================================
class EWSNodeModel(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 4, num_trees: int = 64, 
                 tree_depth: int = 6, signature_depth: int = 3, use_projection: bool = False):
        """
        :param input_dim: 原始输入特征维度 F
        :param latent_dim: B 算子投影到的潜空间维度 (仅在 use_projection=True 时生效)
        :param use_projection: 是否开启 B 算子开关，默认关闭 (False)
        """
        super().__init__()
        self.ews = EWSLayer(
            input_dim=input_dim, 
            latent_dim=latent_dim, 
            depth=signature_depth, 
            use_projection=use_projection
        )
        self.node = DenseODT(
            input_dim=self.ews.output_dim, 
            num_trees=num_trees, 
            depth=tree_depth, 
            output_dim=1
        )
        
    def forward(self, x):
        ews_features = self.ews(x)
        output = self.node(ews_features)
        return output