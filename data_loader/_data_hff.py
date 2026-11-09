import os
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view


from .__utils__ import TARGET_PATH, _ensure_dir

def build_Daily_hff_data_set(
    lookback: int,
    data_path: os.PathLike | str = "data_raw/hff_v1/factors_panel.parquet"
) -> None:

    stock_data = pd.read_parquet(data_path).dropna()

    ftlist = stock_data.columns

    print("正在进行全局数据透视 (Pivot)...")
    # 2. 一次性将所有数据展开 (Unstack)。
    # 这一步会将 (DataDate, SecuCode) 展开，列变成 MultiIndex(特征, 股票)
    pivoted = stock_data.unstack(level="SecuCode")
    

    # 提取维度元数据
    # 注意: 使用 get_level_values().unique() 而非 levels[1]
    # levels[1] 会包含 dropna() 后已不存在的股票（未使用的类别）
    dates = pivoted.index.values
    stocks = pivoted.columns.get_level_values(1).unique().values
    num_dates = len(dates)
    num_features = len(ftlist)
    num_stocks = len(stocks)

    print("正在构建内存视图...")
    # 3. 将底层数据转为 3D NumPy 数组: (Dates, Features, Stocks)
    # Pandas unstack 后的 values 结构正好对应这种 reshape 顺序
    data_3d = pivoted.values.reshape((num_dates, num_features, num_stocks))
    
    # 将形状转换为: (Dates, Stocks, Features)
    data_3d = data_3d.transpose((0, 2, 1))

    # 4. 核心魔法：使用 sliding_window_view 创建零拷贝的滑动窗口
    # 沿着日期轴(axis=0)滑动，窗口大小为 lookback
    # 生成的形状为: (有效天数, Stocks, Features, lookback)
    windows = sliding_window_view(data_3d, window_shape=lookback, axis=0)

    # 我们需要的结果形状为每支股票 (lookback, Features)
    # 因此将后两维转置，最终形状变为: (有效天数, Stocks, lookback, Features)
    windows = windows.transpose((0, 1, 3, 2))
    
    # 计算有效的日期区间（前 lookback-1 天无法形成完整窗口）
    valid_dates = dates[lookback - 1:]
    
    result_idx = []

    print("开始生成并保存切片...")
    # 5. 遍历时间切片并过滤/保存 (此时所有的特征拼装已经完成)
    for idx, day in enumerate(tqdm(valid_dates)):
        # 直接获取当天的 3D 矩阵: 形状 (Stocks, lookback, Features)
        ft = windows[idx]

        # 过滤条件: 切片内不能有 NaN
        cond = np.isnan(ft).sum(axis=(1, 2)) == 0
        ft_filtered = ft[cond]
        valid_stocks = stocks[cond]
        
        if len(ft_filtered) == 0:
            tqdm.write(f'no valid stock after filtering on day {day}')
            continue

        # 保存 .npy
        out_dir = os.path.join(TARGET_PATH, f'hff_1d_lag{lookback}')
        _ensure_dir(out_dir)
        day_str = str(np.datetime_as_string(day, unit='D')).replace('-', '')
        np.save(os.path.join(out_dir, f'{day_str}.npy'), ft_filtered)

        # 收集 metadata (用 list append 比在循环里建 DataFrame 快得多)
        result_idx.extend([(day_str, s) for s in valid_stocks])

    result = pd.DataFrame(result_idx, columns=['DataDate', 'SecuCode'])
    result.to_feather(os.path.join(TARGET_PATH, f'hff_1d_lag{lookback}.feather'))

def load_ft_hff(
    day: str, 
    lookback: int,
    freq: Literal['half_hour', 'half_day', '1d']
) -> np.ndarray:
    """读取已保存的hff特征"""
    return np.load(
        os.path.join(TARGET_PATH, f'hff_{freq}_lag{lookback}', f'{day}.npy'),
        mmap_mode='r+'
    )