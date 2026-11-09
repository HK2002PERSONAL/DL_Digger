from utils.Logger import smart_log, make_log_file; make_log_file()
from utils.DeviceManager import get_free_device
from utils.SeedManager import seed_all_non_torch, seed_torch_all
################################################################
############################ Relying ###########################
################################################################
import os
import argparse

import torch
from torch import nn
from tqdm import tqdm
import pandas as pd

from data_loader.data_set_loader import get_1x1y_dataset
from data_loader.__utils__ import TARGET_PATH
from models.utils import (SimuRankICLoss, ZscoreICLoss,
                          TrainModel, z_score,
                          IdxDataset, 
                          GroupBatchSampler, BatchNormalizerWrapper)
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
class AttAttAtt_Model(nn.Module):

    def __init__(
        self, 
        feature_dim: int,
        seq_len: int,
        num_heads: int = 2,
        dropout: float = 0.2,
        hidden_size: int = 64,
    ) -> None:
        super(AttAttAtt_Model, self).__init__()

        self.hidden_size = hidden_size
        # Linear Position Encoding (1, T, 1) - 0号特征
        self.register_buffer('pos', torch.linspace(0, 1, seq_len).view(1, -1, 1))
        # Causal Mask (T, T) - 上三角为True表示禁止attend未来
        self.register_buffer('causal_mask', torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1))
        # Linear Feature Down Sample (B, T, F+1) -> (B, T, H)，+1是位置编码
        self.ft_in = nn.Linear(in_features=feature_dim + 1, out_features=hidden_size)
        # Time Series Attention (B, T, H) -> (B, T, H)
        self.pre_t_attn_norm = nn.LayerNorm(hidden_size)
        self.t_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.t_attn_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.t_attn_norm = nn.LayerNorm(hidden_size)
        # Asset attention (B, T, H) -> (B, T, H)
        self.pre_asset_attn_norm = nn.LayerNorm(hidden_size)
        self.asset_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, dropout=dropout)
        self.asset_attn_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.asset_attn_norm = nn.LayerNorm(hidden_size)
        # Time Series Attention Pooling
        self.pre_tp_attn_norm = nn.LayerNorm(hidden_size)
        self.tp_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.tp_attn_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.tp_attn_norm = nn.LayerNorm(hidden_size)
        # Decode Out
        self.a_out = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x: torch.Tensor):
        # 位置编码作为0号特征拼接: (B,T,1) cat (B,T,F) -> (B,T,F+1)
        pos = self.pos.expand(x.size(0), -1, -1)
        ft = self.ft_in(torch.cat([pos, x], dim=-1)) # (B, T, F+1) -> (B, T, H)

        ft_t_attn, _ = self.t_attn(
            self.pre_t_attn_norm(ft), self.pre_t_attn_norm(ft), self.pre_t_attn_norm(ft), 
            attn_mask=self.causal_mask) # (B, T, H) -> (B, T, H)
        ft_t_attn = ft + self.t_attn_ffn(self.t_attn_norm(ft + ft_t_attn)) # (B, T, H) -> (B, T, H)

        ft_ast_attn, _ = self.asset_attn(
            self.pre_asset_attn_norm(ft_t_attn), 
            self.pre_asset_attn_norm(ft_t_attn), 
            self.pre_asset_attn_norm(ft_t_attn)
        ) # (B, T, H) -> (B, T, H)
        ft_ast_attn = ft_t_attn + self.asset_attn_ffn(self.asset_attn_norm(ft_t_attn + ft_ast_attn)) # (B, T, H) -> (B, T, H)

        # 只取最后时间步作query: causal mask下最后token可attend全部位置，故无需mask
        ft_out, _ = self.tp_attn(
            self.pre_tp_attn_norm(ft_ast_attn[:, -1:, :]), 
            self.pre_tp_attn_norm(ft_ast_attn), 
            self.pre_tp_attn_norm(ft_ast_attn)
        ) # (B,1,H)x(B,T,H) -> (B, 1, H)
        ft_out = (ft_ast_attn[:, -1:, :] + self.tp_attn_ffn(self.tp_attn_norm(ft_ast_attn[:, -1:, :] + ft_out))).squeeze(1)

        out = self.a_out(ft_out) # (B, H) -> (B, 1)
        return out.squeeze(-1)  # (B, 1) -> (B)
    
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
    train_done: bool = False,
    model_name: str = 'AttAttAtt',
):
    
    smart_log(f"""
{model_name} {vER}
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
        criterion = SimuRankICLoss()
    elif z_score_type == 'batch':
        criterion = ZscoreICLoss()
    else:
        criterion = nn.MSELoss()

    model_path = os.path.join(TARGET_PATH, 'test_model', f'{model_name}{vER}_{ft_type}_{freq}_{lookback}_{label_type}_{label_span}')

    if not train_done:
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        for iter_seed in torch_seed_range:

            model = AttAttAtt_Model(feat_num, lookback).to(device)
            if z_score_type == 'batch':
                model = BatchNormalizerWrapper(model, feat_num).to(device)
            
            model_mark = f'model_{iter_seed}.pt'

            if model_mark in os.listdir(model_path): continue

            smart_log(f'Iter seed: {iter_seed}')

            seed_torch_all(int(iter_seed * 10 + 2), False)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            Main = TrainModel(model, train_loader, valid_loader, criterion, optimizer, model_path, 'IC', optim_gamma=0.99, patience=10, num_epochs=100)

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

    modelS = {iter_seed: AttAttAtt_Model(feat_num, lookback).to(device) for iter_seed in torch_seed_range}
    if z_score_type == 'batch':
        modelS = {iter_seed: BatchNormalizerWrapper(model, feat_num).to(device) for iter_seed, model in modelS.items()}

    for model_num in torch_seed_range:
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
                for model_num in torch_seed_range
            ]), dim=0).cpu().detach().numpy()

            resultS[test_daylist[i]] = pd.Series(batch_pred, index=batch_id)

    resultS = pd.concat(resultS).to_frame('pred')
    resultS.index.names = ['DataDate', 'SecuCode']
    resultS.index = pd.MultiIndex.from_arrays([
        pd.to_datetime(resultS.index.get_level_values('DataDate'), format="%Y%m%d"),
        resultS.index.get_level_values('SecuCode')
    ], names=['DataDate', 'SecuCode'])
    resultS.reset_index(drop=False).to_parquet(
        os.path.join(TARGET_PATH, 'test_result', f'{model_name}{vER}_{freq}_{lookback}_{label_type}_{label_span}.parquet'))


################################################################
############################ Running ###########################
################################################################

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--ft_type', type=str, default="kline", 
                        choices=['kline', 'sig'],
                        help='特征类型')
    parser.add_argument('--feat_num', type=int, default=6, help='特征数量')
    parser.add_argument('--freq', type=str, default="1d",
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
                vER="FFNRank"
            )


