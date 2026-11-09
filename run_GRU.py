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
import numpy as np
import lightgbm as lgb

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
########################## ML Utilities #########################
################################################################
def cross_sectional_rank(s):
    """
    将收益率转换为截面百分位排名 (0 到 1 之间)。
    相同的收益率取平均排名，避免排序噪音。
    """
    return s.rank(pct=True).values

def custom_ic_eval(preds, train_data):
    """
    自定义 LightGBM 评估函数：计算预测值与真实 Label 的 Pearson IC。
    由于我们在外部已经把 Label 转换为了每日截面 Rank，
    所以这里的全局 Pearson IC 在数学上非常接近每日 RankIC 的均值。
    """
    labels = train_data.get_label()
    # 避免标准差为 0 导致的除零错误
    if np.std(preds) <= 1e-8 or np.std(labels) <= 1e-8:
        return 'IC', 0.0, True
    
    # 计算相关系数
    ic = np.corrcoef(preds, labels)[0, 1]
    return 'IC', float(ic), True  # True 表示指标越大越好

def z_score_series(s):
    """对 Series 做 z-score 标准化"""
    return (s - s.mean()) / (s.std() + 1e-8)
################################################################
########################## ML Utilities #########################
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
class Min_Model(nn.Module):
    def __init__(self, feature_dim, lookback=20, time_tag=24, bias=True):
        super(Min_Model, self).__init__()
        self.Min_to_Day_Module = MinDayRNN(feature_dim, lookback, time_tag, bias=bias)
        self.hid_size = self.Min_to_Day_Module.hid_size
        self.Day_to_Ret_Module = BaseRNN(self.hid_size, num_layers=1, bias=bias)

    def forward(self, xm):
        m_rep = self.Min_to_Day_Module(xm)
        return self.Day_to_Ret_Module(m_rep)

class Feature_Model(Min_Model):

    def forward(self, xm):
        m_rep = self.Min_to_Day_Module(xm)
        ft_rep = self.Day_to_Ret_Module.net(m_rep)
        rnn_out, _ = self.Day_to_Ret_Module.rnn(ft_rep) 
        last_hidden = rnn_out[:, -1, :]
        return last_hidden

time_tag_spec = {'1d': 1, 'one_hour': 4, 'half_hour': 8, 'five_minute': 48, 'one_minute': 240}
    
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
BenchMark {vER}
基准模型，双层GRU，分钟频为采样GRU，其他测试的默认模型
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

    model_path = os.path.join(TARGET_PATH, 'test_model', f'GRU{vER}_{ft_type}_{freq}_{lookback}_{label_type}_{label_span}')

    if not train_done:
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        for iter_seed in torch_seed_range:

            model = Min_Model(feat_num, lookback, time_tag_spec[freq]).to(device)
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

    test_idx = test_idx[test_idx.index.get_level_values('SecuCode')!='']
    test_idx = test_idx[~test_idx.index.get_level_values('SecuCode').str.contains('\x00')]

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

    modelS = {iter_seed: Min_Model(feat_num, lookback, time_tag_spec[freq]).to(device) for iter_seed in torch_seed_range}
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
    resultS.reset_index(drop=False).to_parquet(os.path.join(TARGET_PATH, 'test_result', f'GRU{vER}_{freq}_{lookback}_{label_type}_{label_span}.parquet'))

def ml_after_train(
    device: torch.device, 
    ft_type: str,
    feat_num: int, 
    freq: str, 
    lookback: int, 
    label_type: str, 
    label_span: int, 
    vER: str = '',
):
    smart_log(f"""
ML after train {vER}
基准模型，双层GRU，分钟频为采样GRU，其他测试的默认模型
data param: {freq}: {lookback}; {label_type}: {label_span}; {z_score_type}
输出模型 lightgbm
""")

    model_path = os.path.join(TARGET_PATH, 'test_model', f'GRU{vER}_{ft_type}_{freq}_{lookback}_{label_type}_{label_span}')

    (train_x_data, _, train_idx, 
     valid_x_data, _, valid_idx,
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
    
    train_loader = torch.utils.data.DataLoader(
        dataset=IdxDataset(
            train_x_data, 
            train_idx.index.get_level_values('SecuCode').astype(int).values, 
            train_idx['origin_idx'].values), 
        sampler=GroupBatchSampler(
            train_idx.index.get_level_values('DataDate').strftime("%Y%m%d").astype(int).tolist(), 
            None, shuffle=False), 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True)

    train_daylist = train_loader.sampler.group_ids_unique
    
    valid_loader = torch.utils.data.DataLoader(
        dataset=IdxDataset(
            valid_x_data, 
            valid_idx.index.get_level_values('SecuCode').astype(int).values, 
            valid_idx['origin_idx'].values), 
        sampler=GroupBatchSampler(
            valid_idx.index.get_level_values('DataDate').strftime("%Y%m%d").astype(int).tolist(), 
            None, shuffle=False), 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True)

    valid_daylist = valid_loader.sampler.group_ids_unique

    # test_idx = test_idx[test_idx.index.get_level_values('SecuCode')!='']
    # test_idx = test_idx[~test_idx.index.get_level_values('SecuCode').str.contains('\x00')]

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

    # 读取标签（仅读取一次，三个种子共用）
    label = pd.read_feather(os.path.join(TARGET_PATH, 'label', f'{label_type}_ret.ftr'))
    # SecuCode 转 int 与特征索引对齐；索引顺序与特征一致 (DataDate, SecuCode)
    label['SecuCode'] = label['SecuCode'].astype(int)
    label = label.set_index(['DataDate', 'SecuCode']).sort_index()
    smart_log(f'Label loaded: {len(label)} rows')

    for model_num in torch_seed_range:
        
        model = Feature_Model(feat_num, lookback, time_tag_spec[freq]).to(device)
        if z_score_type == 'batch':
            model = BatchNormalizerWrapper(model, feat_num).to(device)

        model.load_state_dict(torch.load( # pylint: disable=used-before-assignment
            os.path.join(model_path, f'model_{model_num}.pt'),
            map_location=device))
        model.eval()

        train_feat, valid_feat, test_feat = {}, {}, {}

        with torch.no_grad():
            # output train feature
            for i, (batch_x, batch_id) in enumerate(tqdm(train_loader, desc='train outputing', leave=False)):

                batch_x = batch_x.to(device).float().squeeze(0).squeeze(1)
                batch_id = batch_id.reshape(-1).numpy()

                train_feat[train_daylist[i]] = pd.DataFrame(
                    model(batch_x).cpu().detach().numpy(), 
                    index=batch_id, columns=range(64))
            # output valid feature
            for i, (batch_x, batch_id) in enumerate(tqdm(valid_loader, desc='valid outputing', leave=False)):

                batch_x = batch_x.to(device).float().squeeze(0).squeeze(1)
                batch_id = batch_id.reshape(-1).numpy()

                valid_feat[valid_daylist[i]] = pd.DataFrame(
                    model(batch_x).cpu().detach().numpy(), 
                    index=batch_id, columns=range(64))
            # output test feature
            for i, (batch_x, batch_id) in enumerate(tqdm(test_loader, desc='test outputing', leave=False)):

                batch_x = batch_x.to(device).float().squeeze(0).squeeze(1)
                batch_id = batch_id.reshape(-1).numpy()

                test_feat[test_daylist[i]] = pd.DataFrame(
                    model(batch_x).cpu().detach().numpy(), 
                    index=batch_id, columns=range(64))

        train_feat = pd.concat(train_feat)
        valid_feat = pd.concat(valid_feat)
        test_feat = pd.concat(test_feat)

        train_feat.index.names = ['DataDate', 'SecuCode']
        valid_feat.index.names = ['DataDate', 'SecuCode']
        test_feat.index.names = ['DataDate', 'SecuCode']

        train_feat.index = pd.MultiIndex.from_arrays([
            pd.to_datetime(train_feat.index.get_level_values('DataDate'), format="%Y%m%d"),
            train_feat.index.get_level_values('SecuCode')
        ], names=['DataDate', 'SecuCode'])

        valid_feat.index = pd.MultiIndex.from_arrays([
            pd.to_datetime(valid_feat.index.get_level_values('DataDate'), format="%Y%m%d"),
            valid_feat.index.get_level_values('SecuCode')
        ], names=['DataDate', 'SecuCode'])

        test_feat.index = pd.MultiIndex.from_arrays([
            pd.to_datetime(test_feat.index.get_level_values('DataDate'), format="%Y%m%d"),
            test_feat.index.get_level_values('SecuCode')
        ], names=['DataDate', 'SecuCode'])

        # train lightgbm
        smart_log(f'Training LightGBM for model seed {model_num}')
        
        # 对齐特征与标签索引
        avail_train_idx = train_feat.index.intersection(label.index)
        avail_valid_idx = valid_feat.index.intersection(label.index)
        
        train_X = train_feat.loc[avail_train_idx]
        valid_X = valid_feat.loc[avail_valid_idx]
        
        train_Y = label.loc[avail_train_idx, f'Return{label_span}d'].dropna()
        valid_Y = label.loc[avail_valid_idx, f'Return{label_span}d'].dropna()
        
        # 重新对齐（确保索引一致）
        common_train_idx = train_X.index.intersection(train_Y.index)
        common_valid_idx = valid_X.index.intersection(valid_Y.index)
        train_X, train_Y = train_X.loc[common_train_idx], train_Y.loc[common_train_idx]
        valid_X, valid_Y = valid_X.loc[common_valid_idx], valid_Y.loc[common_valid_idx]
        
        smart_log(f'Train samples: {len(train_X)}, Valid samples: {len(valid_X)}')
        
        # 标签转截面排名（与 pipeline_ML_base.py 一致）
        train_l = train_Y.groupby(level='DataDate').transform(cross_sectional_rank).values
        valid_l = valid_Y.groupby(level='DataDate').transform(cross_sectional_rank).values
        
        # 构建 LightGBM Dataset
        feature_names = [f'feat_{i}' for i in range(train_feat.shape[1])]
        train_data_lgb = lgb.Dataset(
            train_X.values,
            label=train_l,
            feature_name=feature_names,
            free_raw_data=False
        )
        valid_data_lgb = lgb.Dataset(
            valid_X.values,
            label=valid_l,
            feature_name=feature_names,
            reference=train_data_lgb,
            free_raw_data=False
        )
        
        # 训练参数（参考 pipeline_ML_base.py）
        params = {
            'objective': 'regression',
            'metric': 'None',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': -1,
            'min_data_in_leaf': 100,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'verbose': -1,
            'seed': model_num,
            'n_jobs': 16
        }
        
        # 训练模型
        model_lgb = lgb.train(
            params,
            train_data_lgb,
            num_boost_round=1500,
            valid_sets=[valid_data_lgb],
            valid_names=['valid'],
            feval=custom_ic_eval,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        )
        
        smart_log(f'LightGBM best iteration: {model_lgb.best_iteration}')
        
        # 测试集预测
        test_pred = model_lgb.predict(test_feat.values)
        test_pred_df = pd.DataFrame({'pred': test_pred}, index=test_feat.index)
        
        # 按天 z_score 并累加到总预测
        test_pred_zscore = test_pred_df.groupby(level='DataDate').transform(z_score_series)
        
        if model_num == list(torch_seed_range)[0]:
            all_test_pred = test_pred_zscore.copy()
        else:
            all_test_pred = all_test_pred.add(test_pred_zscore, fill_value=0)
        
        smart_log(f'LightGBM for seed {model_num} done')

    # 保存最终结果
    result_path = os.path.join(
        TARGET_PATH, 'test_result', 
        f'GRU{vER}_{ft_type}_{freq}_{lookback}_{label_type}_{label_span}_MLaT.parquet'
    )
    all_test_pred.reset_index(drop=False).to_parquet(result_path)
    smart_log(f'All LightGBM predictions saved to {result_path}')

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

            # main(
            #     device=test_device,
            #     ft_type=args.ft_type,
            #     feat_num=args.feat_num,
            #     freq=args.freq,
            #     lookback=args.lookback,
            #     label_type=_label_type,
            #     label_span=_label_span,
            #     vER="IC"
            # )

            ml_after_train(
                device=test_device,
                ft_type=args.ft_type,
                feat_num=args.feat_num,
                freq=args.freq,
                lookback=args.lookback,
                label_type=_label_type,
                label_span=_label_span,
                vER="IC"
            )


