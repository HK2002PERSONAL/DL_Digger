from utils.Logger import smart_log, make_log_file; make_log_file()
from utils.DeviceManager import get_free_device
from utils.SeedManager import seed_all_non_torch, seed_torch_all
################################################################
############################ Relying ###########################
################################################################
import os
import argparse

import signatory
import torch
from torch import nn
from tqdm import tqdm
import pandas as pd

from data_loader.data_set_loader import get_1x1y_dataset
from data_loader.__utils__ import TARGET_PATH
from models.utils import (ICLoss, ZscoreICLoss,
                          TrainModel, z_score,
                          IdxDataset, 
                          GroupBatchSampler, BatchNormalizerWrapper)
from models.RNN import BaseRNN, MinDayRNN
################################################################
############################ Relying ###########################
################################################################

################################################################
########################## Train Param #########################
################################################################
train_day_begin = 20140101
train_day_end = 20191201
valid_day_begin = 20200101
valid_day_end = 20211201
test_data_begin = 20220101
test_data_end = 20270101
z_score_type = 'set'

root_seed = 10086
torch_seed_range = range(root_seed, root_seed + 10, 4)
################################################################
########################## Train Param #########################
################################################################

################################################################
######################## Model Structure #######################
################################################################
class GRU_SigAtt_Model(nn.Module):

    def __init__(
        self, 
        # alpha model args
        feature_dim: int,
        lookback: int,
        time_tag: int,
        # asset attention args
        num_heads: int = 1,
        dropout: float = 0.1,
        hidden_size: int = 64,
    ) -> None:
        super(GRU_SigAtt_Model, self).__init__()
        
        # GRU 特征提取
        self.gru_ft_emb = MinDayRNN(feature_dim, lookback, time_tag, hidden_size=hidden_size, bias=True)
        self.hidden_size = hidden_size
        
        # Asset attention (简化版，无 Horizon 维度)
        # 输入形状: (T, D, hidden_size) - T=时间步, D=资产数
        self.asset_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)

        # 解码器
        self.gru_dec_alpha = BaseRNN(d_feat=hidden_size * 2, hidden_size=hidden_size, num_layers=1, bias=True)

    def _cal_cross_sig(self, x: torch.Tensor, price_mark: int = 3) -> torch.Tensor:
        B = x.shape[0]
        asset_path = x[:,:,price_mark]
        norm_asset_path = (asset_path - asset_path.mean(dim=-1, keepdim=True)) / (asset_path.std(dim=-1, keepdim=True) + 1e-8)
        norm_asset_path = torch.clip(norm_asset_path, -5, 5).transpose(0,1) # (B, H) -> (H,B)
        sig = signatory.signature(norm_asset_path.unsqueeze(0), depth=2).squeeze(0) # (B + B^2)
        cross_sig = sig[B:].reshape((B, B))
        return cross_sig

    def forward(self, x: torch.Tensor):
        # x: (D, seq_len, feature_dim) - D=资产数, seq_len=时间长度, feature_dim=特征维度

        # 1. GRU 特征提取
        m_rep = self.gru_ft_emb(x)  # (D, T, hidden_size) - T=seq_len//time_tag

        # 2. Asset attention (在资产维度 D 上做注意力)
        # 对每个时间步独立做 asset attention
        # D, T, M = m_rep.shape
        x_ast, _ = self.asset_attn(m_rep, m_rep, m_rep)#, attn_mask=cross_sig) # (D, T, hidden_size)

        # 3. 时序解码
        out = self.gru_dec_alpha(torch.concat([m_rep, x_ast], dim=-1))  # (D, 1)

        return out.squeeze(-1)  # (D,)

time_tag_spec = {'1d': 1, 'one_hour': 4, 'half_hour': 8, 'five_minute': 48, 'one_minute': 240}

class SimuRankICLoss(ICLoss):

    def forward(self, x, y):
        y = torch.argsort(torch.argsort(y)).reshape(-1) / (len(y) - 1)
        return super().forward(x, y)
    
################################################################
######################## Model Structure #######################
################################################################

################################################################
############################ Running ###########################
################################################################
def main(
    device: torch.device, 
    ft_type: str,
    feat_num: int, 
    freq: str, 
    lookback: int, 
    label_type: str, 
    label_span: int, 
    vER: str = '',
    train_done: bool = False
):
    
    smart_log(f"""
GRU SigAtt {vER}
data param: {freq}: {lookback}; {label_type}: {label_span}; {z_score_type}
""")

    (train_x_data, train_y_data, train_idx, 
     valid_x_data, valid_y_data, valid_idx,
     test_x_data, test_idx) = get_1x1y_dataset(
        train_day_begin=train_day_begin,
        train_day_end=train_day_end,
        valid_day_begin=valid_day_begin,
        valid_day_end=valid_day_end,
        test_day_begin=test_data_begin,
        test_day_end=test_data_end,
        lookback=lookback,
        ft_type=ft_type,
        freq=freq,
        device=None,
        max_worker=32,
        label_type=label_type,
        label_span=label_span,
        z_score_type=z_score_type
    )
    
    smart_log('assemblying data')

    train_group_mark = train_idx.index.get_level_values('DataDate').strftime("%Y%m%d").astype(int).tolist()
    valid_group_mark = valid_idx.index.get_level_values('DataDate').strftime("%Y%m%d").astype(int).tolist()
    
    train_loader = torch.utils.data.DataLoader(
        dataset=IdxDataset(train_x_data, train_y_data, train_idx['origin_idx'].values), 
        sampler=GroupBatchSampler(train_group_mark, None), 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True)
    
    valid_loader = torch.utils.data.DataLoader(
        dataset=IdxDataset(valid_x_data, valid_y_data, valid_idx['origin_idx'].values), 
        sampler=GroupBatchSampler(valid_group_mark, None), 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True)
    
    if z_score_type == 'set':
        criterion = ICLoss()
    elif z_score_type == 'batch':
        criterion = ZscoreICLoss()
    else:
        criterion = nn.MSELoss()

    model_path = os.path.join(TARGET_PATH, 'test_model', f'GruAttResGru{vER}_{ft_type}_{freq}_{lookback}_{label_type}_{label_span}')

    if not train_done:
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        for iter_seed in torch_seed_range:

            model = GRU_SigAtt_Model(feat_num, lookback, time_tag_spec[freq]).to(device)
            if z_score_type == 'batch':
                model = BatchNormalizerWrapper(model, feat_num).to(device)
            
            model_mark = f'model_{iter_seed}.pt'

            if model_mark in os.listdir(model_path): continue

            smart_log(f'Iter seed: {iter_seed}')

            seed_torch_all(int(iter_seed * 10 + 2), False)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            Main = TrainModel(model, train_loader, valid_loader, criterion, optimizer, model_path, optim_gamma=0.99, patience=10, num_epochs=100)

            Main.train_model()

            torch.save(Main.model.state_dict(), os.path.join(model_path, model_mark))

        del train_loader, valid_loader
        torch.cuda.empty_cache()

    test_loader = torch.utils.data.DataLoader(
        dataset=IdxDataset(
            test_x_data, 
            test_idx.index.get_level_values('SecuCode').astype(int).values, 
            test_idx['origin_idx'].values), 
        sampler=GroupBatchSampler(
            test_idx.index.get_level_values('DataDate').strftime("%Y%m%d").astype(int).tolist(), 
            None, shuffle=False), 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True)

    test_daylist = test_loader.sampler.group_ids_unique

    _torch_seed_range = torch_seed_range

    modelS = {iter_seed: GRU_SigAtt_Model(feat_num, lookback, time_tag_spec[freq]).to(device) for iter_seed in _torch_seed_range}
    if z_score_type == 'batch':
        modelS = {iter_seed: BatchNormalizerWrapper(model, feat_num).to(device) for iter_seed, model in modelS.items()}

    for model_num in _torch_seed_range:
        modelS[model_num].load_state_dict(torch.load( # pylint: disable=used-before-assignment
            os.path.join(model_path, f'model_{model_num}.pt'),
            map_location=device))
        modelS[model_num].eval()

    resultS = {}

    with torch.no_grad():
        for i, (batch_x, batch_id) in enumerate(tqdm(test_loader, desc='outputing', leave=False)):

            batch_x = batch_x.to(device).float().squeeze(0).squeeze(1)
            batch_id = batch_id.reshape(-1).numpy()

            batch_pred = torch.sum(torch.stack([
                z_score(modelS[model_num](batch_x).reshape(-1))
                for model_num in _torch_seed_range
            ]), dim=0).cpu().detach().numpy()

            resultS[test_daylist[i]] = pd.Series(batch_pred, index=batch_id)

    resultS = pd.concat(resultS).to_frame('pred')
    resultS.index.names = ['DataDate', 'SecuCode']
    resultS.index = pd.MultiIndex.from_arrays([
        pd.to_datetime(resultS.index.get_level_values('DataDate'), format="%Y%m%d"),
        resultS.index.get_level_values('SecuCode')
    ], names=['DataDate', 'SecuCode'])
    resultS.reset_index(drop=False).to_parquet(os.path.join(TARGET_PATH, 'test_result', f'GruAttResGru{vER}_{freq}_{lookback}_{label_type}_{label_span}.parquet'))


################################################################
############################ Running ###########################
################################################################

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='GRU Model Training')
    parser.add_argument('--ft_type', type=str, default="kline", 
                        choices=['kline', 'sig'],
                        help='特征类型')
    parser.add_argument('--feat_num', type=int, default=6, help='特征数量')
    parser.add_argument('--freq', type=str, default="1d",
                        choices=['1d', 'one_hour', 'half_hour', 'five_minute', 'one_minute'],
                        help='数据频率 (example: one_minute)')
    parser.add_argument('--lookback', type=int, default=80, help='回看长度 (example: 2)注意需要与freq匹配')
    args = parser.parse_args()

    seed_all_non_torch(root_seed)

    test_device = get_free_device()

    for _label_type in ['spec']:
        for _label_span in [5]:
            smart_log(f"{_label_type}_{_label_span}")

            main(
                device=test_device,
                ft_type=args.ft_type,
                feat_num=args.feat_num,
                freq=args.freq,
                lookback=args.lookback,
                label_type=_label_type,
                label_span=_label_span,
                vER="IC"
            )


