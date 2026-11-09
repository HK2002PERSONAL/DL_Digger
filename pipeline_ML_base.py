import os
import re
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
from pandas.tseries.offsets import BDay

from factory.back_testV2 import batch_test

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

def preprocess_column_index(_data):
    # 强制统一 Index 结构，避免 level 顺序不一致
    if 'DataDate' in _data.columns and 'SecuCode' in _data.columns:
        _data = _data.set_index(['DataDate', 'SecuCode'])
    _data = _data.sort_index()

    # 处理列名
    if isinstance(_data.columns, pd.MultiIndex):
        _data.columns = ['_'.join(col).strip('_') for col in _data.columns.values]
    else:
        _data.columns = [str(c) for c in _data.columns]

    return _data

# 清洗列名：tuple 拍平 + 非法字符替换
def clean_name(col):
    # tuple 拍平
    if isinstance(col, tuple):
        col = '_'.join(str(c) for c in col)
    # 把所有非字母/数字/下划线字符替换为 _
    col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
    # 去掉连续下划线
    col = re.sub(r'_+', '_', col).strip('_')
    return col

def main_loop(
    data_set: str = 'TechPureETE',
    label_type: str = '',
    label_span: int = 5):

    VERSION = f'{data_set}_{label_type}r{label_span}'
    print(VERSION)
    # ============================================
    # 1. 数据读取 & 对齐
    # ============================================
    print("Loading data...")
    data = preprocess_column_index(pd.read_feather(os.path.join("factor_feature", f"{data_set}.ftr")))
    data.index = data.index.swaplevel()
    data.sort_index(inplace=True)

    label_type_mark = {'':'stock', 'd':'debarra', 's':'spec', 'c':'stock_clean'}[label_type]
    label = pd.read_feather(os.path.join("data", "label", f"{label_type_mark}_ret.ftr"))

    label = label.set_index(['DataDate', 'SecuCode']).sort_index()

    avail_idx = data.index.intersection(label.index)
    data = data.loc[avail_idx]
    label = label.loc[avail_idx, f'Return{label_span}d'].dropna() #####################################

    feature_names = list([clean_name(c) for c in data.columns])

    # 提取年份用于滚动划分
    mark_year = pd.Series(
        data.index.get_level_values('DataDate').year,
        index=data.index
    )

    forward_year = pd.Series(
        (data.index.get_level_values('DataDate') + BDay(label_span)).year,
        index=data.index
    )

    predictions = []
    ic_records = []

    # ============================================
    # 2. 滚动训练
    # ============================================
    for test_year in range(2016, 2027):

        print(f"\n========== Training for year {test_year} ==========")

        valid_year = test_year - 1

        # 划分训练 / 验证 / 测试
        train_mask = mark_year < valid_year
        valid_mask = (mark_year == valid_year) & (forward_year <= valid_year)
        test_mask = mark_year == test_year

        train_X = data.loc[train_mask]
        train_Y = label.loc[train_mask]

        valid_X = data.loc[valid_mask]
        valid_Y = label.loc[valid_mask]

        test_X = data.loc[test_mask]
        test_Y = label.loc[test_mask]

        # 数据量不足则跳过该年（例如第一年可能没有足够的历史数据）
        if len(train_X) == 0 or len(test_X) == 0:
            print(f"Skipping {test_year} due to insufficient data.")
            continue

        # ============================================
        # Label 处理：截面 Rank 编码 (0~1)
        # ============================================
        train_l = train_Y.groupby(level='DataDate').transform(cross_sectional_rank).values
        valid_l = valid_Y.groupby(level='DataDate').transform(cross_sectional_rank).values

        # ============================================
        # LGBM Dataset (无需 group 参数)
        # ============================================
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

        # ============================================
        # 训练参数 (Regression)
        # ============================================
        params = {
            'objective': 'regression',
            'metric': 'None',                 # 禁用默认 metric，完全依赖我们自定义的 IC
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': -1,
            'min_data_in_leaf': 100,          # 相比 Lambdarank，回归防过拟合叶子节点可以设大一点
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'verbose': -1,
            'seed': 10086,
            'n_jobs': 16
        }

        # 训练并根据 valid 的 IC 进行早停
        model = lgb.train(
            params,
            train_data_lgb,
            num_boost_round=1500,             
            valid_sets=[train_data_lgb, valid_data_lgb],
            valid_names=['train', 'valid'],
            feval=custom_ic_eval,             # 传入自定义的 IC 评估函数
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=50) # 每50轮打印一次指标
            ]
        )

        # ============================================
        # 3. 测试集预测与评估
        # ============================================
        test_pred = model.predict(test_X.values)

        pred_df = pd.DataFrame({
            'pred': test_pred,
            'label': test_Y.values          # 注意这里存回了【原始收益率】，用于计算真实的策略 IC
        }, index=test_X.index)
        
        pred_df['year'] = test_year
        predictions.append(pred_df)

        # 每日截面 Pearson IC
        daily_ic = pred_df.groupby(level='DataDate').apply(
            lambda df: df['pred'].corr(df['label'], method='pearson')
        ).dropna()

        # 每日截面 Spearman RankIC
        daily_rankic = pred_df.groupby(level='DataDate').apply(
            lambda df: df['pred'].corr(df['label'], method='spearman')
        ).dropna()

        # 年度汇总统计
        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std(ddof=1)
        rankic_mean = daily_rankic.mean()
        rankic_std = daily_rankic.std(ddof=1)

        ic_ir = ic_mean / ic_std if ic_std > 1e-8 else np.nan
        rankic_ir = rankic_mean / rankic_std if rankic_std > 1e-8 else np.nan

        best_iter = model.best_iteration

        ic_records.append({
            'year': test_year,
            'IC_mean': ic_mean,
            'IC_std': ic_std,
            'IC_IR': ic_ir,
            'RankIC_mean': rankic_mean,
            'RankIC_std': rankic_std,
            'RankIC_IR': rankic_ir,
            'n_days': len(daily_ic),
            'best_iteration': best_iter
        })

        print(f"[{test_year}] IC={ic_mean:.4f} (IR={ic_ir:.3f}) | "
              f"RankIC={rankic_mean:.4f} (RankIR={rankic_ir:.3f}) | "
              f"best_iter={best_iter}")

    # ============================================
    # 4. 汇总保存
    # ============================================
    if len(predictions) > 0:
        os.makedirs(F"factor_selection/V{VERSION}", exist_ok=True)
        
        all_pred = pd.concat(predictions)
        all_pred[['pred']].reset_index().to_parquet(
            os.path.join("factor_selection", F"V{VERSION}", "Tech_LGBM_Regression_Rank_pred.parquet")
        )

        ic_df = pd.DataFrame(ic_records)
        ic_df.to_csv(
            os.path.join("factor_selection", F"V{VERSION}", "Tech_LGBM_Regression_IC_stats.csv"),
            index=False
        )
        print("\nAll tasks finished. Results saved.")
    else:
        print("\nNo predictions were generated.")

    return F"V{VERSION}"


if __name__ == '__main__':
    
    for _span in [5, 20]:

        test_mark = main_loop('TechPureETE_coverage_test', 'c', _span)

        signal_data = pd.read_parquet(f"factor_selection/{test_mark}/Tech_LGBM_Regression_Rank_pred.parquet")
        signal_data = signal_data.set_index(['SecuCode', 'DataDate']).sort_index()

        for smooth_decay in [5, 10, 15, 20]:
            signal_data[f'pred_EWM{smooth_decay}'] = signal_data.groupby("SecuCode")['pred'].ewm(span=smooth_decay).mean().droplevel(0)

        signal_data = signal_data[signal_data.index.get_level_values('DataDate') > pd.to_datetime("2019-12-31")]

        batch_test(signal_data, test_mark, 
            neutral_factor=['beta', 'bp', 'ey', 'growth', 'leverage', 'liquidity', 'lncap', 'momentum', 'nls', 'rv', 'contrarian', 'short_rev'])