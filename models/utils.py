import random
from collections import defaultdict

import numpy as np
# import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Sampler

from utils.Logger import smart_log


# ============================================================
#                      评价 / 损失函数
# ============================================================

class LongShortRet(nn.Module):
    """
    计算多空收益：
    - 从预测 x 中取 top-q 做多，bottom-q 做空
    - 返回 (long_ret, short_ret, long_short_ret)

    参数
    ----
    q : float
        头尾分位比例
    absolulte_Ret : bool
        若 False，则先对 y 做去均值
    """
    def __init__(self, q: float = 0.1, absolulte_Ret: bool = False):
        super().__init__()
        self.q = q
        self.absolute_Ret = absolulte_Ret

    def forward(self, x, y, mcls=None, position: float = 1.0):
        x = x.reshape(-1).float()
        y = y.reshape(-1).float()

        if not self.absolute_Ret:
            y = y - y.mean()

        B = x.numel()
        k = max(1, int(B * self.q))

        long_idx = torch.topk(x, k, largest=True).indices
        short_idx = torch.topk(x, k, largest=False).indices

        long_ret = y[long_idx].mean()
        short_ret = (-y[short_idx]).mean()
        long_short_ret = long_ret + short_ret

        # 可选：根据 mcls 动态调整仓位
        if mcls is not None:
            position = mcls[long_idx].mean()
            if position > 0.6:
                position = 1.0
            elif position < 0.4:
                position = 0.0
            else:
                position = 0.5

        return long_ret.item() * position, short_ret.item(), long_short_ret.item()


class LongIdx(nn.Module):
    """
    返回 top-q 的样本索引（按 x 排序）。
    """
    def __init__(self, q: float = 0.1):
        super().__init__()
        self.q = q

    def forward(self, x):
        x = x.reshape(-1).float()

        B = x.numel()
        k = max(1, int(B * self.q))
        long_idx = torch.topk(x, k, largest=True).indices

        return long_idx.cpu().detach().numpy()


class LongRetLoss(nn.Module):
    """
    注意：当前实现与 LongIdx 等价，返回 top-q 索引。
    （保持你的原始行为，不改业务逻辑）
    """
    def __init__(self, q: float = 0.1):
        super().__init__()
        self.q = q

    def forward(self, x):
        x = x.reshape(-1).float()

        B = x.numel()
        k = max(1, int(B * self.q))
        long_idx = torch.topk(x, k, largest=True).indices

        return long_idx.cpu().detach().numpy()


class ICLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        x = x.reshape(-1).float()
        y = y.reshape(-1).float()
        xm = x - x.mean()
        ym = y - y.mean()
        denom = (xm.norm(p=2) * ym.norm(p=2)).clamp_min(self.eps)  # ← 关键保护
        return -xm.dot(ym) / denom


class LongBiasICLoss(ICLoss):

    def __init__(self, eps: float = 1e-8, negative_slope: float = 0.1):
        super().__init__(eps)
        self.bias_actor = nn.LeakyReLU(negative_slope)

    def forward(self, x, y):
        x = x.reshape(-1).float()
        y = y.reshape(-1).float()
        xm = self.bias_actor(x - x.mean())
        ym = self.bias_actor(y - y.mean())
        return - xm.dot(ym) / (xm.norm(p=2) * ym.norm(p=2)).clamp_min(self.eps)

class SimuRankICLoss(ICLoss):

    def forward(self, x, y):
        y = torch.argsort(torch.argsort(y)).reshape(-1) / (len(y) - 1)
        return super().forward(x, y)


class SoftRankICLoss(nn.Module):
    """
    可导 RankIC（Spearman 近似）：
      rank_ic ≈ corr(soft_rank(x), soft_rank(y))
      loss = -rank_ic
    """
    def __init__(self, tau: float = 1.0, eps: float = 1e-12, corr_type: str = 'pearson', sign_type: str = 'sigmoid'):
        super().__init__()
        self.tau = tau
        self.eps = eps
        self.corr_type = corr_type
        self.sign_type = sign_type

    def soft_rank_pairwise(self, x: torch.Tensor, tau: float = 1.0, include_self: bool = False) -> torch.Tensor:
        x = x.reshape(-1).float()
        diff = (x[None, :] - x[:, None]) / tau
        P = getattr(torch, self.sign_type)(diff)

        if not include_self:
            P = P - torch.diag(torch.diag(P))

        r = 1.0 + P.sum(dim=1)
        return r

    def pearson_corr(self, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        a = a.reshape(-1).float()
        b = b.reshape(-1).float()
        am = a - a.mean()
        bm = b - b.mean()
        denom = (am.norm(p=2) * bm.norm(p=2)).clamp_min(eps)
        return am.dot(bm) / denom

    def spearman_corr(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.reshape(-1).float()
        b = b.reshape(-1).float()
        n = a.shape[0]
        return 1 - 6 * torch.sum(torch.square(a - b)) / (n * (n * n - 1))

    def forward(self, x, y):
        x = z_score(x).reshape(-1).float()
        y = y.reshape(-1).float()

        rx = self.soft_rank_pairwise(x, tau=self.tau, include_self=False)
        ry = self.soft_rank_pairwise(y, tau=self.tau, include_self=False)

        if self.corr_type == 'spearman':
            rank_ic = self.spearman_corr(rx, ry)
        elif self.corr_type == 'pearson':
            rank_ic = self.pearson_corr(rx, ry, eps=self.eps)
        else:
            raise ValueError(f'Unsupported corr_type: {self.corr_type}')

        return -rank_ic


class ZscoreICLoss(nn.Module):
    """
    先对 y 做 z-score，再计算 -corr(x, y)。
    """
    def __init__(self):
        super(ZscoreICLoss, self).__init__()

    def forward(self, x, y: torch.Tensor):
        x = x.reshape(-1).float()
        y = y.reshape(-1).float()
        y = y.sub(torch.mean(y)).div(torch.std(y)).clip(-5, 5)
        xm = x.sub(torch.mean(x))
        ym = y.sub(torch.mean(y))
        icloss = xm.dot(ym) / (torch.norm(xm, 2) * torch.norm(ym, 2))
        return -icloss


class RankICLoss(nn.Module):
    """
    离散 rank 的 RankIC loss。
    """
    def __init__(self):
        super(RankICLoss, self).__init__()

    def forward(self, x: torch.Tensor, y):
        x = x.reshape(-1).float()
        y = y.reshape(-1).float()
        x_rank = torch.argsort(torch.argsort(x)).float()
        y_rank = torch.argsort(torch.argsort(y)).float()

        n = x.shape[0]
        rankicloss = 1 - 6 * torch.sum(torch.square(x_rank - y_rank)) / (n * (n * n - 1))
        return -rankicloss


def z_score(y: torch.Tensor) -> torch.Tensor:
    """标准 z-score"""
    return (y - y.mean()) / (y.std() + 1e-8)


def robust_zscore(x: torch.Tensor) -> torch.Tensor:
    """robust z-score（q10/q50/q90）"""
    q10, x_mean, q90 = x.quantile(torch.tensor([0.1, 0.5, 0.9]).to(x.device))
    return x.subtract(x_mean).divide(q90 - q10 + 1e-12).clip(-5, 5)


# ============================================================
#                      EarlyStopping
# ============================================================

class EarlyStopping:
    """
    早停机制：验证损失在 patience 轮内无提升则停止。
    """
    def __init__(self, save_path, patience=10, verbose=False, delta=0):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            smart_log(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        验证损失下降时保存模型参数。
        """
        if self.verbose:
            smart_log(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.save_path + '/checkpoint.pt')
        self.val_loss_min = val_loss


# ============================================================
#                      Sampler / Dataset
# ============================================================

class GroupBatchSampler(Sampler):
    """
    按 group 采样：
    - batch_size 不为 None：普通随机小批次
    - batch_size 为 None：按 group 整组输出
    """
    def __init__(self, group_ids, batch_size, shuffle: bool = True):
        self.group_ids = group_ids
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.group_idx = defaultdict(list)
        for idx, group_id in enumerate(group_ids):
            self.group_idx[group_id].append(idx)

        self.group_ids_unique = list(self.group_idx.keys())

        if self.shuffle:
            random.shuffle(self.group_ids_unique)

    def __iter__(self):
        if self.batch_size is not None:
            idxS = list(range(len(self.group_ids)))
            random.shuffle(idxS)
            for start_id in range(0, len(idxS), self.batch_size):
                yield idxS[start_id:start_id + self.batch_size]
        else:
            for group_id in self.group_ids_unique:
                yield self.group_idx[group_id]

    def __len__(self):
        if self.batch_size is not None:
            return len(self.group_ids) // self.batch_size + 1
        else:
            return len(self.group_ids_unique)


class BaseDataset(torch.utils.data.Dataset):
    """
    基础 dataset：统一 numpy / tensor 输入处理。
    """
    def __init__(self, x, y):
        self.x = self.__check_and_from_numpy__(x)
        self.y = self.__check_and_from_numpy__(y)

    def __check_and_from_numpy__(self, dt):
        if torch.is_tensor(dt):
            return dt
        elif isinstance(dt, np.ndarray):
            return torch.from_numpy(dt)
        else:
            raise ValueError(f'unsupported input {type(dt)}')

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


class IdxDataset(BaseDataset):
    """
    根据 idx 从共享特征池中取样。
    """
    def __init__(self, x, y, idx):
        super().__init__(x, y)
        self.idx = idx

    def __getitem__(self, index):
        return self.x[self.idx[index]], self.y[index]


# ============================================================
#                      训练器
# ============================================================

class TrainModel:
    """
    通用训练器：
    - train/valid 循环
    - 学习率调度
    - EarlyStopping
    """
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, save_path, inspection='IC', **kwargs):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_path = save_path
        self.kwargs = kwargs
        self.device = next(model.parameters()).device
        self.inspection = inspection
        if inspection == 'IC':
            self.inspectionloss = ICLoss()
        elif inspection == 'RankIC':
            self.inspectionloss = RankICLoss()
        else:
            raise ValueError(f"{inspection} inspection loss not supported yet")

    def train(self, verbose: int):
        self.model.train()
        train_loss = 0.0
        train_loss_ic = 0.0

        if verbose > 1:
            loss_tracker = torch.empty(size=(len(self.train_loader),), dtype=torch.float32, device='cpu')
            loss_ic_tracker = torch.empty(size=(len(self.train_loader),), dtype=torch.float32, device='cpu')

        for batch_idx, d in enumerate(tqdm(self.train_loader, desc='training', leave=False)):
            x = d[0].to(self.device).float()
            y = d[-1].to(self.device).float()

            x = x.squeeze(0).squeeze(1)
            y = y.squeeze(0)

            output = self.model(x)
            loss = self.criterion(output, y)
            icloss = self.inspectionloss(output, y)

            train_loss += loss.item()
            train_loss_ic += icloss.item()

            if verbose > 0:
                if batch_idx % 10 == 5:
                    tqdm.write(f'{train_loss / batch_idx}, {train_loss_ic / batch_idx}')
                if verbose > 1:
                    loss_tracker[batch_idx] = loss.item()
                    loss_ic_tracker[batch_idx] = icloss.item()

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        train_loss /= len(self.train_loader)
        train_loss_ic /= len(self.train_loader)

        if verbose > 1:
            print(torch.sum(torch.isnan(loss_tracker)), torch.mean(loss_tracker), torch.max(loss_tracker), torch.min(loss_tracker))
            print(torch.sum(torch.isnan(loss_ic_tracker)), torch.mean(loss_ic_tracker), torch.max(loss_ic_tracker), torch.min(loss_ic_tracker))

        return train_loss, train_loss_ic

    def valid(self, verbose: int):
        self.model.eval()
        valid_loss = 0.0
        valid_loss_ic = 0.0

        if verbose > 1:
            loss_tracker = torch.empty(size=(len(self.train_loader),), dtype=torch.float32, device='cpu')
            loss_ic_tracker = torch.empty(size=(len(self.train_loader),), dtype=torch.float32, device='cpu')

        with torch.no_grad():
            for batch_idx, d in enumerate(tqdm(self.valid_loader, desc='validating', leave=False)):
                x = d[0].to(self.device).float()
                y = d[-1].to(self.device).float()

                x = x.squeeze(0).squeeze(1)
                y = y.squeeze(0)

                output = self.model(x)
                loss = self.criterion(output, y)
                icloss = self.inspectionloss(output, y)

                valid_loss += loss.item()
                valid_loss_ic += icloss.item()

                if verbose > 0:
                    if batch_idx % 200 == 100:
                        tqdm.write(f'{valid_loss / batch_idx}, {valid_loss_ic / batch_idx}')
                    if verbose > 1:
                        loss_tracker[batch_idx] = loss.item()
                        loss_ic_tracker[batch_idx] = icloss.item()

        valid_loss /= len(self.valid_loader)
        valid_loss_ic /= len(self.valid_loader)

        if verbose > 1:
            print(torch.sum(torch.isnan(loss_tracker)), torch.mean(loss_tracker), torch.max(loss_tracker), torch.min(loss_tracker))
            print(torch.sum(torch.isnan(loss_ic_tracker)), torch.mean(loss_ic_tracker), torch.max(loss_ic_tracker), torch.min(loss_ic_tracker))

        return valid_loss, valid_loss_ic

    def train_model(self, verbose: int = 0):
        """
        启动训练流程并返回最佳验证分数（负号已还原）。
        """
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.kwargs['optim_gamma'])
        early_stopping = EarlyStopping(
            save_path=self.save_path,
            patience=self.kwargs['patience'],
            verbose=True
        )

        train_losses = []
        valid_losses = []

        for epoch in range(self.kwargs['num_epochs']):
            train_loss, train_loss_ic = self.train(verbose)
            scheduler.step()
            valid_loss, valid_loss_ic = self.valid(verbose)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            epoch_len = len(str(self.kwargs['num_epochs']))
            num_epochs = self.kwargs['num_epochs']

            print_msg = (
                f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] '
                f'train_loss: {train_loss:.5f} '
                f'train_loss_{self.inspection}: {train_loss_ic:.5f} '
                f'valid_loss: {valid_loss:.5f} '
                f'valid_loss_{self.inspection}: {valid_loss_ic:.5f}'
            )
            smart_log(print_msg)

            early_stopping(valid_loss, self.model)
            if early_stopping.early_stop:
                smart_log(f"Early stopping at: {-early_stopping.best_score}")
                break

        # 加载最佳 checkpoint
        self.model.load_state_dict(torch.load(self.save_path + '/checkpoint.pt'))

        return -early_stopping.best_score


class AmpTrainModel(TrainModel):
    """
    AMP 混合精度训练器（4090 / Ada Lovelace 优化）：
    - 继承 TrainModel，使用 torch.autocast 进行混合精度训练
    - 默认使用 bfloat16（动态范围同 float32，数值稳定，无需梯度缩放）
    - 可显著减少显存占用（约 40-50%）
    """
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, save_path, inspection='IC', amp_dtype=torch.bfloat16, **kwargs):
        super().__init__(model, train_loader, valid_loader, criterion, optimizer, save_path, inspection, **kwargs)
        self.amp_dtype = amp_dtype
        smart_log(f"using amp type: {self.amp_dtype}")
        # bfloat16 无需 GradScaler；float16 需要
        self.scaler = torch.cuda.amp.GradScaler() if amp_dtype == torch.float16 else None

    def train(self, verbose: int):
        self.model.train()
        train_loss = 0.0
        train_loss_ic = 0.0

        if verbose > 1:
            loss_tracker = torch.empty(size=(len(self.train_loader),), dtype=torch.float32, device='cpu')
            loss_ic_tracker = torch.empty(size=(len(self.train_loader),), dtype=torch.float32, device='cpu')

        for batch_idx, d in enumerate(tqdm(self.train_loader, desc='training', leave=False)):
            x = d[0].to(self.device).float()
            y = d[-1].to(self.device).float()

            x = x.squeeze(0).squeeze(1)
            y = y.squeeze(0)

            self.optimizer.zero_grad()

            # AMP 前向传播
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
                output = self.model(x)
                loss = self.criterion(output, y)
                icloss = self.inspectionloss(output, y)

            train_loss += loss.item()
            train_loss_ic += icloss.item()

            if verbose > 0:
                if batch_idx % 10 == 5:
                    tqdm.write(f'{train_loss / (batch_idx + 1)}, {train_loss_ic / (batch_idx + 1)}')
                if verbose > 1:
                    loss_tracker[batch_idx] = loss.item()
                    loss_ic_tracker[batch_idx] = icloss.item()

            # 反向传播
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        train_loss /= len(self.train_loader)
        train_loss_ic /= len(self.train_loader)

        if verbose > 1:
            print(torch.sum(torch.isnan(loss_tracker)), torch.mean(loss_tracker), torch.max(loss_tracker), torch.min(loss_tracker))
            print(torch.sum(torch.isnan(loss_ic_tracker)), torch.mean(loss_ic_tracker), torch.max(loss_ic_tracker), torch.min(loss_ic_tracker))

        return train_loss, train_loss_ic

    def valid(self, verbose: int):
        self.model.eval()
        valid_loss = 0.0
        valid_loss_ic = 0.0

        if verbose > 1:
            loss_tracker = torch.empty(size=(len(self.valid_loader),), dtype=torch.float32, device='cpu')
            loss_ic_tracker = torch.empty(size=(len(self.valid_loader),), dtype=torch.float32, device='cpu')

        with torch.no_grad():
            for batch_idx, d in enumerate(tqdm(self.valid_loader, desc='validating', leave=False)):
                x = d[0].to(self.device).float()
                y = d[-1].to(self.device).float()

                x = x.squeeze(0).squeeze(1)
                y = y.squeeze(0)

                # AMP 前向传播
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
                    output = self.model(x)

                loss = self.criterion(output, y)
                icloss = self.inspectionloss(output, y)

                valid_loss += loss.item()
                valid_loss_ic += icloss.item()

                if verbose > 0:
                    if batch_idx % 200 == 100:
                        tqdm.write(f'{valid_loss / (batch_idx + 1)}, {valid_loss_ic / (batch_idx + 1)}')
                    if verbose > 1:
                        loss_tracker[batch_idx] = loss.item()
                        loss_ic_tracker[batch_idx] = icloss.item()

        valid_loss /= len(self.valid_loader)
        valid_loss_ic /= len(self.valid_loader)

        if verbose > 1:
            print(torch.sum(torch.isnan(loss_tracker)), torch.mean(loss_tracker), torch.max(loss_tracker), torch.min(loss_tracker))
            print(torch.sum(torch.isnan(loss_ic_tracker)), torch.mean(loss_ic_tracker), torch.max(loss_ic_tracker), torch.min(loss_ic_tracker))

        return valid_loss, valid_loss_ic


# ============================================================
#                      GSAM 优化器
# ============================================================

class GSAM(torch.optim.Optimizer):
    """
    Surrogate Gap Minimization (GSAM) 优化器。
    
    参考: Zhuang et al., "Surrogate Gap Minimization Improves Sharpness-Aware Training", ICLR 2022
    
    GSAM 相比 SAM 的改进：
    1. 最小化代理间隙（surrogate gap）而非直接最小化扰动点损失
    2. 通过梯度差异惩罚项提升训练稳定性
    3. 更新公式: w = w - lr * (∇L(w) + α * (∇L(w') - ∇L(w)))
       其中 w' = w + ε 是扰动点，α 是代理间隙系数
    
    参数
    ----
    params : iterable
        待优化参数
    base_optimizer : torch.optim.Optimizer 类
        基础优化器（如 torch.optim.Adam, torch.optim.SGD）
    rho : float
        扰动半径，控制邻域大小（默认 0.05）
    alpha : float
        代理间隙系数，控制梯度差异惩罚权重（默认 0.5）
    adaptive : bool
        是否使用自适应缩放（按参数绝对值缩放扰动）
    **kwargs :
        传递给基础优化器的其他参数（如 lr, weight_decay）
    """
    def __init__(self, params, base_optimizer, rho=0.05, alpha=0.5, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho value: {rho}, should be non-negative"
        assert alpha >= 0.0, f"Invalid alpha value: {alpha}, should be non-negative"
        
        defaults = dict(rho=rho, alpha=alpha, adaptive=adaptive)
        defaults.update(kwargs)
        super(GSAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        第一步：保存原始梯度，计算梯度范数，将参数推进到扰动点 w' = w + ε
        """
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 保存原始参数和原始梯度
                self.state[p]['old_p'] = p.data.clone()
                self.state[p]['old_grad'] = p.grad.clone()
                
                # 自适应缩放：按参数绝对值缩放扰动
                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * p.grad * scale.to(p)
                
                # 推进到扰动点
                p.add_(e_w)
        
        if zero_grad:
            self.zero_grad()
            
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        第二步：计算代理间隙梯度，恢复原始参数，执行更新
        
        更新公式: w = w - lr * (∇L(w) + α * (∇L(w') - ∇L(w)))
        等价于: w = w - lr * ((1-α) * ∇L(w) + α * ∇L(w'))
        """
        for group in self.param_groups:
            alpha = group['alpha']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 获取原始梯度（扰动前）
                old_grad = self.state[p]['old_grad']
                # 当前梯度是扰动点的梯度
                perturbed_grad = p.grad.clone()
                
                # 计算 GSAM 组合梯度: (1-α) * ∇L(w) + α * ∇L(w')
                # 等价于: ∇L(w) + α * (∇L(w') - ∇L(w))
                combined_grad = old_grad + alpha * (perturbed_grad - old_grad)
                
                # 恢复原始参数
                p.data = self.state[p]['old_p']
                
                # 设置组合梯度用于更新
                p.grad = combined_grad
        
        # 用组合梯度执行基础优化器更新
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
            
    @torch.no_grad()
    def step(self, closure=None):
        """
        完整 GSAM 步骤（需要 closure 重新计算 loss）：
        1. 第一次前向/反向 -> first_step
        2. 第二次前向/反向（在扰动点）-> second_step
        """
        assert closure is not None, "GSAM requires closure to recompute loss at perturbed point"
        
        # 第一次前向/反向
        with torch.enable_grad():
            loss = closure()
        self.first_step(zero_grad=True)
        
        # 第二次前向/反向（在扰动点）
        with torch.enable_grad():
            closure()
        self.second_step(zero_grad=True)
        
        return loss
    
    def _grad_norm(self):
        """计算所有参数梯度的全局范数"""
        shared_device = self.param_groups[0]['params'][0].device
        
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group['adaptive'] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups 
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def load_state_dict(self, state_dict):
        """加载状态（委托给基础优化器）"""
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
    
    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none)


class GSAMTrainModel(TrainModel):
    """
    GSAM 优化训练器：
    - 继承 TrainModel，适配 GSAM 两阶段优化
    - 每个 batch 执行两次前向/反向传播
    - 通过代理间隙最小化寻找平坦极小值，提升泛化能力
    
    用法
    ----
    # 方式1：传入 GSAM 优化器（推荐）
    base_optimizer = torch.optim.Adam
    optimizer = GSAM(model.parameters(), base_optimizer, rho=0.05, alpha=0.5, lr=1e-3)
    trainer = GSAMTrainModel(model, train_loader, valid_loader, criterion, optimizer, save_path, **kwargs)
    
    # 方式2：传入基础优化器，由 GSAMTrainModel 自动包装
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = GSAMTrainModel(model, train_loader, valid_loader, criterion, optimizer, save_path, rho=0.05, **kwargs)
    """
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, save_path, inspection='IC', **kwargs):
        super().__init__(model, train_loader, valid_loader, criterion, optimizer, save_path, inspection, **kwargs)
        
        # 如果传入的不是 GSAM 优化器，自动包装
        if not isinstance(self.optimizer, GSAM):
            rho = kwargs.pop('rho', 0.05)
            alpha = kwargs.pop('alpha', 0.5)
            adaptive = kwargs.pop('adaptive', False)
            
            # 提取基础优化器类型和参数
            base_optimizer_type = type(self.optimizer)
            # 从现有优化器提取参数配置
            optimizer_kwargs = {}
            if self.optimizer.param_groups:
                pg = self.optimizer.param_groups[0]
                for key in ['lr', 'momentum', 'weight_decay', 'betas', 'eps']:
                    if key in pg:
                        optimizer_kwargs[key] = pg[key]
            
            self.optimizer = GSAM(
                model.parameters(), 
                base_optimizer_type, 
                rho=rho, 
                alpha=alpha,
                adaptive=adaptive,
                **optimizer_kwargs
            )
            smart_log(f"GSAMTrainModel: wrapped {base_optimizer_type.__name__} with GSAM (rho={rho}, alpha={alpha}, adaptive={adaptive})")
        else:
            smart_log(f"GSAMTrainModel: using GSAM optimizer (rho={self.optimizer.param_groups[0].get('rho', 'N/A')}, alpha={self.optimizer.param_groups[0].get('alpha', 'N/A')})")
    
    def train(self, verbose: int):
        """
        GSAM 训练循环：两阶段前向/反向传播
        
        阶段1：正常前向 -> 计算 loss -> 反向 -> first_step（扰动参数）
        阶段2：扰动点前向 -> 计算 loss -> 反向 -> second_step（组合梯度并更新）
        """
        self.model.train()
        train_loss = 0.0
        train_loss_ic = 0.0
        
        if verbose > 1:
            loss_tracker = torch.empty(size=(len(self.train_loader),), dtype=torch.float32, device='cpu')
            loss_ic_tracker = torch.empty(size=(len(self.train_loader),), dtype=torch.float32, device='cpu')
        
        for batch_idx, d in enumerate(tqdm(self.train_loader, desc='training (GSAM)', leave=False)):
            x = d[0].to(self.device).float()
            y = d[-1].to(self.device).float()
            
            x = x.squeeze(0).squeeze(1)
            y = y.squeeze(0)
            
            # ==================== 第一阶段 ====================
            # 正常前向传播
            output = self.model(x)
            loss = self.criterion(output, y)
            icloss = self.inspectionloss(output, y)
            
            train_loss += loss.item()
            train_loss_ic += icloss.item()
            
            if verbose > 0:
                if batch_idx % 10 == 5:
                    tqdm.write(f'{train_loss / (batch_idx + 1)}, {train_loss_ic / (batch_idx + 1)}')
                if verbose > 1:
                    loss_tracker[batch_idx] = loss.item()
                    loss_ic_tracker[batch_idx] = icloss.item()
            
            # 反向传播并执行第一步（参数扰动）
            loss.backward()
            self.optimizer.first_step(zero_grad=True)
            
            # ==================== 第二阶段 ====================
            # 在扰动点重新前向传播
            output_2 = self.model(x)
            loss_2 = self.criterion(output_2, y)
            
            # 反向传播并执行第二步（恢复参数并更新）
            loss_2.backward()
            self.optimizer.second_step(zero_grad=True)
        
        train_loss /= len(self.train_loader)
        train_loss_ic /= len(self.train_loader)
        
        if verbose > 1:
            print(torch.sum(torch.isnan(loss_tracker)), torch.mean(loss_tracker), torch.max(loss_tracker), torch.min(loss_tracker))
            print(torch.sum(torch.isnan(loss_ic_tracker)), torch.mean(loss_ic_tracker), torch.max(loss_ic_tracker), torch.min(loss_ic_tracker))
        
        return train_loss, train_loss_ic


# ============================================================
#                 Batch 级输入标准化 Wrapper
# ============================================================

class BatchNormalizerWrapper(nn.Module):
    """
    在 forward 内对输入做 batch 级 robust 归一化，再送入 model。
    """
    def __init__(self, model, clip_range=5.0):
        super().__init__()
        self.model = model
        self.clip_range = clip_range
        self.q = torch.tensor([0.5, 0.1, 0.9])  # median, q10, q90（保持原顺序）

    def forward(self, x: torch.Tensor):
        B, S, F = x.shape
        flat_x = x.reshape(-1, F)

        # 对 volume / turnnum 通道做 log1p
        flat_x[:, 6:8].log1p_()

        x_median, q10, q90 = torch.quantile(
            flat_x,
            self.q.to(device=flat_x.device, dtype=flat_x.dtype),
            dim=0,
            keepdim=True
        )

        flat_x.sub_(x_median).div_((q90 - q10) + 1E-12)
        flat_x.clamp_(-self.clip_range, self.clip_range)

        return self.model(flat_x.view(B, S, F))