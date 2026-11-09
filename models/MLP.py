"""
MLP 模型：Linear + SiLU + Linear + SiLU + Linear

提供两种调用方式：
1. sklearn 风格：MLPRegressor(...).fit(X, y).predict(X)
2. LightGBM 风格：Dataset + train() 函数，可无缝替换 lgb.Dataset / lgb.train
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin


# ============================================================
#                      网络结构
# ============================================================
class MLPNet(nn.Module):
    """
    Linear + SiLU + Linear + SiLU + Linear
    
    Parameters
    ----------
    in_features : int
        输入特征维度
    hidden_features : int or list[int]
        隐藏层维度。若为 int，则两个隐藏层维度相同；
        若为 list，则分别为两个隐藏层的维度。
    out_features : int
        输出维度，默认 1
    """
    def __init__(self, in_features: int, hidden_features=256, out_features: int = 1):
        super().__init__()
        if isinstance(hidden_features, int):
            hidden_features = [hidden_features, hidden_features]
        
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features[0]),
            nn.SiLU(),
            nn.Linear(hidden_features[0], hidden_features[1]),
            nn.SiLU(),
            nn.Linear(hidden_features[1], out_features),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
#                      sklearn 风格封装
# ============================================================
class MLPRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn 风格的 MLP 回归器，支持 CUDA。
    
    Parameters
    ----------
    hidden_features : int or list[int], default=256
        隐藏层维度
    lr : float, default=0.001
        学习率
    epochs : int, default=100
        训练轮数
    batch_size : int, default=4096
        批大小
    device : str or torch.device, default=None
        设备，默认自动选择 (cuda if available else cpu)
    seed : int, default=None
        随机种子
    verbose : bool, default=False
        是否打印训练日志
    """
    def __init__(self, hidden_features=256, lr=0.001, epochs=100, 
                 batch_size=4096, device=None, seed=None, verbose=False):
        self.hidden_features = hidden_features
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.verbose = verbose
        
        self.model_ = None
        self.device_ = None
        self.in_features_ = None
    
    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, eval_metric=None):
        """
        训练模型。
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        eval_set : list of (X, y) tuples, optional
            验证集，用于早停
        early_stopping_rounds : int, optional
            早停轮数
        eval_metric : callable, optional
            评估函数，签名: metric(y_true, y_pred) -> float (越大越好)
            
        Returns
        -------
        self
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        
        self.device_ = self._get_device(); print(f"Using {self.device_}")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.in_features_ = X.shape[1]
        
        self.model_ = MLPNet(self.in_features_, self.hidden_features).to(self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        train_dataset = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 验证集准备
        valid_loader = None
        if eval_set is not None and len(eval_set) > 0:
            vX, vy = eval_set[0]
            vX = np.asarray(vX, dtype=np.float32)
            vy = np.asarray(vy, dtype=np.float32)
            valid_dataset = TensorDataset(torch.from_numpy(vX), torch.from_numpy(vy))
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        
        best_score = -np.inf
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            # Train
            self.model_.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device_), yb.to(self.device_)
                pred = self.model_(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(train_dataset)
            
            # Valid
            if valid_loader is not None and eval_metric is not None:
                self.model_.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for xb, yb in valid_loader:
                        xb = xb.to(self.device_)
                        pred = self.model_(xb)
                        all_preds.append(pred.cpu().numpy())
                        all_labels.append(yb.numpy())
                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                score = eval_metric(all_labels, all_preds)
                
                if self.verbose:
                    print(f"[{epoch+1}/{self.epochs}] loss: {epoch_loss:.6f}, valid_score: {score:.6f}")
                
                if score > best_score:
                    best_score = score
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                else:
                    patience_counter += 1
                    if early_stopping_rounds is not None and patience_counter >= early_stopping_rounds:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch+1}, best score: {best_score:.6f}")
                        break
            else:
                if self.verbose:
                    print(f"[{epoch+1}/{self.epochs}] loss: {epoch_loss:.6f}")
        
        # 恢复最佳模型
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        
        return self
    
    def predict(self, X):
        """
        预测。
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model_.eval()
        all_preds = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device_)
                pred = self.model_(xb)
                all_preds.append(pred.cpu().numpy())
        
        return np.concatenate(all_preds)


# ============================================================
#                LightGBM 风格接口（无缝替换）
# ============================================================
class Dataset:
    """
    类似 lgb.Dataset 的数据封装。
    
    Parameters
    ----------
    data : array-like
        特征数据
    label : array-like, optional
        标签
    feature_name : list[str], optional
        特征名
    reference : Dataset, optional
        参考数据集（兼容 lgb.Dataset 接口，此处不使用）
    free_raw_data : bool, optional
        兼容参数
    """
    def __init__(self, data, label=None, feature_name=None, reference=None, free_raw_data=True):
        self.data = np.asarray(data, dtype=np.float32)
        self.label = np.asarray(label, dtype=np.float32) if label is not None else None
        self.feature_name = feature_name
        self.reference = reference
        self.free_raw_data = free_raw_data
    
    def get_label(self):
        return self.label
    
    def get_data(self):
        return self.data
    
    def num_data(self):
        return len(self.data)
    
    def num_feature(self):
        return self.data.shape[1] if self.data.ndim > 1 else 1


class Booster:
    """
    类似 lgb.Booster 的模型封装。
    """
    def __init__(self, model: MLPNet, device: torch.device):
        self.model = model
        self.device = device
        self.best_iteration = None
    
    def predict(self, X, num_iteration=None):
        """
        预测。
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
        """
        X = np.asarray(X, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=8192, shuffle=False)
        
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                pred = self.model(xb)
                all_preds.append(pred.cpu().numpy())
        
        return np.concatenate(all_preds)


def train(
    params: dict,
    train_set: Dataset,
    num_boost_round: int = 100,
    valid_sets: list = None,
    valid_names: list = None,
    feval=None,
    callbacks: list = None,
    batch_size: int = None,
):
    """
    类似 lgb.train 的训练函数。
    
    Parameters
    ----------
    params : dict
        训练参数，支持的键：
        - 'hidden_features': int or list[int], default=256
        - 'learning_rate' or 'lr': float, default=0.001
        - 'epochs': int, default=num_boost_round
        - 'batch_size': int, default=4096
        - 'device': str or torch.device, default=None (auto)
        - 'seed': int, default=None
        - 'verbose': bool, default=False
        其他 LightGBM 参数会被忽略（兼容用途）。
    train_set : Dataset
        训练数据集
    num_boost_round : int
        训练轮数（对应 epochs）
    valid_sets : list[Dataset], optional
        验证集列表
    valid_names : list[str], optional
        验证集名称
    feval : callable, optional
        自定义评估函数，签名: feval(preds, train_data) -> (name, value, is_higher_better)
        与 LightGBM 的 feval 签名一致。
    callbacks : list, optional
        回调函数列表（兼容用途，暂不实现）
    batch_size : int, optional
        批大小，优先级高于 params
    
    Returns
    -------
    Booster
        训练好的模型
    """
    # 解析参数
    hidden_features = params.get('hidden_features', 256)
    lr = params.get('learning_rate', params.get('lr', 0.001))
    epochs = params.get('epochs', num_boost_round)
    batch_size = batch_size or params.get('batch_size', 4096)
    device_str = params.get('device', None)
    seed = params.get('seed', None)
    verbose = params.get('verbose', True)
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # 设备
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据
    X = train_set.get_data()
    y = train_set.get_label()
    in_features = X.shape[1] if X.ndim > 1 else 1
    
    # 模型
    model = MLPNet(in_features, hidden_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 验证集
    valid_loader = None
    if valid_sets is not None and len(valid_sets) > 0:
        vset = valid_sets[0]
        vX = vset.get_data()
        vy = vset.get_label()
        valid_dataset = TensorDataset(torch.from_numpy(vX), torch.from_numpy(vy))
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    best_score = -np.inf
    best_state = None
    best_iteration = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_dataset)
        
        # Valid
        if valid_loader is not None and feval is not None:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb = xb.to(device)
                    pred = model(xb)
                    all_preds.append(pred.cpu().numpy())
                    all_labels.append(yb.numpy())
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            
            # 调用 feval（LightGBM 风格）
            if valid_sets is not None:
                eval_result = feval(all_preds, valid_sets[0])
                if isinstance(eval_result, tuple) and len(eval_result) == 3:
                    _, score, is_higher_better = eval_result
                else:
                    score = eval_result
                    is_higher_better = True
            else:
                score = 0
                is_higher_better = True
            
            if verbose:
                vname = valid_names[0] if valid_names else 'valid'
                print(f"[{epoch+1}/{epochs}] train_loss: {epoch_loss:.6f}, {vname}_score: {score:.6f}")
            
            if is_higher_better:
                if score > best_score:
                    best_score = score
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    best_iteration = epoch + 1
            else:
                if score < best_score or best_score == -np.inf:
                    best_score = score
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    best_iteration = epoch + 1
        else:
            if verbose:
                print(f"[{epoch+1}/{epochs}] train_loss: {epoch_loss:.6f}")
    
    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
    
    booster = Booster(model, device)
    booster.best_iteration = best_iteration if best_iteration > 0 else epochs
    
    return booster