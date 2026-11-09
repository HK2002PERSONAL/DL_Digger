from utils.Logger import smart_log, make_log_file; make_log_file()
from utils.DeviceManager import get_free_device
from utils.SeedManager import seed_all_non_torch
################################################################
############################ Relying ###########################
################################################################
import os
import argparse

import torch
# from torch import nn
from tqdm import tqdm
import pandas as pd
import numpy as np
import lightgbm as lgb

from data_loader.data_set_loader import get_1x1y_dataset
from data_loader.__utils__ import TARGET_PATH
from models.utils import IdxDataset, GroupBatchSampler, BatchNormalizerWrapper
from models.TCN import TCNModel
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

# 所有频率配置 (freq, lookback)
FREQ_CONFIGS = [
    ('1d', 80),
    # ('half_hour', 40),
    ('five_minute', 10),
    # ('one_minute', 2),
]
################################################################
########################## Train Param #########################
################################################################

################################################################
######################## Model Structure #######################
################################################################
class TCN_Feature_Model(TCNModel):
    """提取 TCN 最后时间步的 64 维特征"""

    def forward(self, x):
        x = x.transpose(1, 2) 
        output = self.network(x)
        last_step_feat = output[:, :, -1]
        return last_step_feat
################################################################
######################## Model Structure #######################
################################################################

################################################################
############################ Running ###########################
################################################################
def ml_after_train_multi_freq(
    device: torch.device, 
    ft_type: str,
    feat_num: int, 
    label_type: str, 
    label_span: int, 
    vER: str = '',
):
    smart_log(f"""
ML after train multi-freq {vER}
多频率特征拼接，TCN 模型，提取最后时间步特征
data param: multi-freq; {label_type}: {label_span}; {z_score_type}
输出模型 lightgbm
""")

    # 读取标签（仅读取一次，所有种子共用）
    label = pd.read_feather(os.path.join(TARGET_PATH, 'label', f'{label_type}_ret.ftr'))
    # SecuCode 转 int 与特征索引对齐；索引顺序与特征一致 (DataDate, SecuCode)
    label['SecuCode'] = label['SecuCode'].astype(int)
    label = label.set_index(['DataDate', 'SecuCode']).sort_index()
    smart_log(f'Label loaded: {len(label)} rows')

    # 收集所有频率的特征
    all_train_feats = {}
    all_valid_feats = {}
    all_test_feats = {}

    for freq, lookback in FREQ_CONFIGS:
        smart_log(f'Loading features for freq={freq}, lookback={lookback}')

        model_path = os.path.join(TARGET_PATH, 'test_model', f'TCN{vER}_{ft_type}_{freq}_{lookback}_{label_type}_{label_span}')

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

        for model_num in torch_seed_range:
            
            model = TCN_Feature_Model(feat_num).to(device)
            if z_score_type == 'batch':
                model = BatchNormalizerWrapper(model, feat_num).to(device)

            model.load_state_dict(torch.load(
                os.path.join(model_path, f'model_{model_num}.pt'),
                map_location=device))
            model.eval()

            train_feat, valid_feat, test_feat = {}, {}, {}

            with torch.no_grad():
                # output train feature
                for i, (batch_x, batch_id) in enumerate(tqdm(train_loader, desc=f'{freq} train outputing', leave=False)):

                    batch_x = batch_x.to(device).float().squeeze(0).squeeze(1)
                    batch_id = batch_id.reshape(-1).numpy()

                    train_feat[train_daylist[i]] = pd.DataFrame(
                        model(batch_x).cpu().detach().numpy(), 
                        index=batch_id, columns=[f'{freq}_{i}' for i in range(64)])
                # output valid feature
                for i, (batch_x, batch_id) in enumerate(tqdm(valid_loader, desc=f'{freq} valid outputing', leave=False)):

                    batch_x = batch_x.to(device).float().squeeze(0).squeeze(1)
                    batch_id = batch_id.reshape(-1).numpy()

                    valid_feat[valid_daylist[i]] = pd.DataFrame(
                        model(batch_x).cpu().detach().numpy(), 
                        index=batch_id, columns=[f'{freq}_{i}' for i in range(64)])
                # output test feature
                for i, (batch_x, batch_id) in enumerate(tqdm(test_loader, desc=f'{freq} test outputing', leave=False)):

                    batch_x = batch_x.to(device).float().squeeze(0).squeeze(1)
                    batch_id = batch_id.reshape(-1).numpy()

                    test_feat[test_daylist[i]] = pd.DataFrame(
                        model(batch_x).cpu().detach().numpy(), 
                        index=batch_id, columns=[f'{freq}_{i}' for i in range(64)])

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

            if model_num not in all_train_feats:
                all_train_feats[model_num] = []
                all_valid_feats[model_num] = []
                all_test_feats[model_num] = []

            all_train_feats[model_num].append(train_feat)
            all_valid_feats[model_num].append(valid_feat)
            all_test_feats[model_num].append(test_feat)

            smart_log(f'Freq {freq} seed {model_num} features extracted')

        smart_log(f'Freq {freq} features extracted')

    # 拼接所有频率的特征并训练
    for model_num in torch_seed_range:
        smart_log(f'Training LightGBM for model seed {model_num}')
        
        # 拼接所有频率的特征（按列拼接）
        train_feat_all = pd.concat(all_train_feats[model_num], axis=1, join="inner")
        valid_feat_all = pd.concat(all_valid_feats[model_num], axis=1, join="inner")
        test_feat_all = pd.concat(all_test_feats[model_num], axis=1, join="inner")
        
        smart_log(f'Combined feature shape: {train_feat_all.shape}')
        
        # 对齐特征与标签索引
        avail_train_idx = train_feat_all.index.intersection(label.index)
        avail_valid_idx = valid_feat_all.index.intersection(label.index)
        
        train_X = train_feat_all.loc[avail_train_idx]
        valid_X = valid_feat_all.loc[avail_valid_idx]
        
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
        feature_names = [f'feat_{i}' for i in range(train_feat_all.shape[1])]
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
            'n_jobs': 24
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
        test_pred = model_lgb.predict(test_feat_all.values)
        test_pred_df = pd.DataFrame({'pred': test_pred}, index=test_feat_all.index)
        
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
        f'TCN{vER}_{ft_type}_[{'_'.join([_item[0] for _item in FREQ_CONFIGS])}]_{label_type}_{label_span}_MLaT.parquet'
    )
    all_test_pred.reset_index(drop=False).to_parquet(result_path)
    smart_log(f'All LightGBM predictions saved to {result_path}')

################################################################
############################ Running ###########################
################################################################

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='TCN Multi-Freq ML Training')
    parser.add_argument('--ft_type', type=str, default="kline", 
                        choices=['kline', 'sig'],
                        help='特征类型')
    parser.add_argument('--feat_num', type=int, default=6, help='特征数量')
    args = parser.parse_args()

    seed_all_non_torch(root_seed)

    test_device = get_free_device()

    for _label_type in ['spec']:
        for _label_span in [5]:
            smart_log(f"{_label_type}_{_label_span}")

            ml_after_train_multi_freq(
                device=test_device,
                ft_type=args.ft_type,
                feat_num=args.feat_num,
                label_type=_label_type,
                label_span=_label_span,
                vER="IC"
            )