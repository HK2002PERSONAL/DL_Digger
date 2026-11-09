import os
from typing import Optional, Literal

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view


from .__utils__ import SOURCE_PATH, TARGET_PATH, _list_days_from_dir, _get_lookback_days, _ensure_dir

freq_map = {"fifteen_minute":15, "half_hour": 30, "one_hour": 60}

agg_rules = {
        "open_price": "first",
        "high_price": "max",
        "low_price": "min",
        "close_price": "last",
        "turnover_volume": "sum",
        "turnover_value": "sum"
    }


def get_intraday_times(freq: Literal['one_minute', 'five_minute', 'half_hour', 'one_hour']):
    step_map = {
        'one_minute': 1,
        'five_minute': 5,
        'half_hour': 30,
        'one_hour': 60
    }
    
    if freq not in step_map:
        raise ValueError(f"Unsupported freq: {freq}. Supported: {list(step_map.keys())}")
    
    step = step_map[freq]
    
    # 上午 9:30 ~ 11:30，起点不含 11:30（11:30 是收盘，不能作为新 bar 起点）
    morning = list(range(9 * 60 + 30, 11 * 60 + 30, step))
    
    # 下午
    if freq == 'one_minute':
        # 1 分钟线特殊：最多到 14:56
        afternoon = list(range(13 * 60, 14 * 60 + 57, step))
    else:
        # 其他频率：起点不含 15:00
        afternoon = list(range(13 * 60, 15 * 60, step))
    
    return [(m // 60) * 100 + (m % 60) for m in morning + afternoon]


def load_kline_min(
    day: str, 
    freq: Literal['one_minute', 'five_minute', 'fifteen_minute', 'half_hour', 'one_hour']
) -> pd.DataFrame:
    """
    读取指定交易日的分钟K线数据。

    参数
    ----
    - day : str 交易日，格式 'YYYYMMDD'。
    - freq : str 数据频率

    返回
    ----
    - pd.DataFrame
        当日分钟数据。
    """
    if freq in freq_map.keys():
        data_root_dir = os.path.join(SOURCE_PATH, "five_minute")
    else:
        data_root_dir = os.path.join(SOURCE_PATH, freq)
        
    d = pd.read_parquet(os.path.join(data_root_dir, f'{day}.parquet')) # pylint: disable=possibly-used-before-assignment

    if freq in freq_map.keys():
        d = d.sort_values(['security_code', 'start_time'])
        d['start_time'] = (d['start_time'] // 100 * 60 + d['start_time'] % 100) // freq_map[freq] * freq_map[freq]
        d = d.groupby(['security_code', 'start_time']).agg(agg_rules)

        level0 = d.index.get_level_values(0)          # security_code
        level1 = d.index.get_level_values(1)          # time_bucket (570, 600, 630...)

        # 重建 MultiIndex
        d.index = pd.MultiIndex.from_arrays(
            [level0, (level1 // 60) * 100 + (level1 % 60)],
            names=['security_code', 'time_bucket']
        )

    else:
        d = d.set_index(['security_code', 'start_time']).sort_index()
        d = d[agg_rules.keys()]
            
    return d

def load_kline_min_hybrid(
    day: str, 
    freq: Literal['one_minute', 'five_minute', 'fifteen_minute', 'half_hour', 'one_hour']
) -> pd.DataFrame:
    """
    读取指定交易日的分钟K线数据。
    - 需重采样时：使用 Polars 提速聚合，最终转回 Pandas MultiIndex。
    - 无需重采样时：直接使用 Pandas 处理。
    """
    # 1. 确定数据根目录
    if freq in freq_map.keys():
        data_root_dir = os.path.join(SOURCE_PATH, "five_minute")
    else:
        data_root_dir = os.path.join(SOURCE_PATH, freq)
        
    # 2. 分支处理
    if freq in freq_map.keys():
        # ======= 【需要重采样】使用 Polars 加速 =======
        f_val = freq_map[freq]
        file_path = os.path.join(data_root_dir, f'{day}.parquet')
        
        # 使用 Polars 读取并聚合
        df_pl = pl.read_parquet(file_path)
        df_pl = (
            df_pl
            # 时间转换（HHMM -> 分钟数 -> 对齐频率）
            .with_columns(
                time_minutes = ((pl.col('start_time') // 100 * 60 + pl.col('start_time') % 100) // f_val * f_val)
            )
            # 分组聚合
            .group_by(['security_code', 'time_minutes'])
            .agg([
                pl.col("open_price").first().alias("open_price"),
                pl.col("high_price").max().alias("high_price"),
                pl.col("low_price").min().alias("low_price"),
                pl.col("close_price").last().alias("close_price"),
                pl.col("turnover_volume").sum().alias("turnover_volume"),
                pl.col("turnover_value").sum().alias("turnover_value"),
            ])
            # 分钟数转回 HHMM 格式
            .with_columns(
                time_bucket = (pl.col('time_minutes') // 60 * 100 + pl.col('time_minutes') % 60)
            )
            # 保持需要的列
            .select(["security_code", "time_bucket", "open_price", "high_price", "low_price", "close_price", "turnover_volume", "turnover_value"])
        )
        
        # 核心：转回 Pandas 并构建与原来一模一样的 MultiIndex
        d = df_pl.to_pandas()
        d = d.sort_values(['security_code', 'time_bucket'])
        d = d.set_index(['security_code', 'time_bucket'])
        
    else:
        # ======= 【无需重采样】纯 Pandas 原生逻辑 =======
        d = pd.read_parquet(os.path.join(data_root_dir, f'{day}.parquet'))
        d = d.set_index(['security_code', 'start_time']).sort_index()
        # 只保留原 agg_rules 的 key 列
        target_cols = ["open_price", "high_price", "low_price", "close_price", "turnover_volume", "turnover_value"]
        d = d[target_cols]
            
    return d

def build_ft_kline_min(
    day: str, 
    lookback: int,
    freq: Literal['one_minute', 'five_minute', 'fifteen_minute', 'half_hour', 'one_hour']
) -> Optional[pd.DataFrame]:
    """
    构建分钟时序特征并保存为 .npy。

    参数
    ----
    day : str
        目标交易日。
    lookback : int
        回看交易日数量。

    返回
    ----
    Optional[pd.DataFrame]
        成功时返回 ['securityid', 'date']，失败返回 None。
    """
    try:
        if freq in freq_map.keys():
            data_root_dir = os.path.join(SOURCE_PATH, "five_minute")
        else:
            data_root_dir = os.path.join(SOURCE_PATH, freq)
        daylist = _list_days_from_dir(data_root_dir)
        daylist_t = _get_lookback_days(day, daylist, lookback)

        if not daylist_t:
            tqdm.write(f'no available day <= {day}')
            return None

        d = pd.concat({d_: load_kline_min_hybrid(d_, freq) for d_ in daylist_t}).sort_index()

        # 非活跃交易填值
        d[["open_price", "high_price", "low_price", "close_price"]] = d[["open_price", "high_price", "low_price", "close_price"]].ffill()
        d[["turnover_volume", "turnover_value"]] = d[["turnover_volume", "turnover_value"]].fillna(0)

        d = d.dropna(how='all').astype(float)

        # 量纲调整
        d[['turnover_volume', 'turnover_value']] = np.log1p(d[['turnover_volume', 'turnover_value']])
        
        times = get_intraday_times(freq)
        full_time_index = pd.MultiIndex.from_product([daylist_t, times])
        stocklist = np.unique(d.index.get_level_values("security_code"))
        ftlist = list(agg_rules.keys())

        ft = np.empty((len(stocklist), lookback * len(times), len(ftlist)))
        for i, ftname in enumerate(ftlist):
            dt = d[ftname].unstack(level=[0,2])
            dt = dt.reindex(index=stocklist, columns=full_time_index)
            ft[:, :, i] = dt.values

        # 过滤条件
        cond1 = np.isnan(ft).sum(axis=(1, 2)) == 0
        cond2 = ft[:, -len(times):, 4].sum(axis=1) > 0
        cond3 = ft[:, -len(times):, 5].sum(axis=1) > 0
        cond = cond1 & cond2 & cond3
        ft = ft[cond, :, :]
        stocklist = [j for i, j in enumerate(stocklist) if cond[i]]

        if len(ft) == 0:
            tqdm.write(f'no valid stock after filtering on day {day}')
            return None

        # 时序标准化
        # 第4列（close_price）做价格归一化
        ft[:, :, 0:4] = ft[:, :, 0:4] / (ft[:, -1, 3].reshape((len(stocklist), 1, 1)) + 1E-12)
        # 第5,6列（volume/value）用最近16个时间步求和归一化
        ft[:, :, 4] = ft[:, :, 4] / (ft[:, -16:, 4].sum(axis=1).reshape((len(stocklist), 1)) + 1E-12)
        ft[:, :, 5] = ft[:, :, 5] / (ft[:, -16:, 5].sum(axis=1).reshape((len(stocklist), 1)) + 1E-12)

        ft[np.isinf(ft)] = np.nan
        cond = np.isnan(ft).sum(axis=(1, 2)) == 0
        ft = ft[cond, :, :]
        stocklist = [j for i, j in enumerate(stocklist) if cond[i]]

        if len(ft) < 10:
            tqdm.write(f'too little data left on day {day}')
            return None

        out_dir = os.path.join(TARGET_PATH, f'kline_{freq}_lag{lookback}')
        _ensure_dir(out_dir)
        np.save(os.path.join(out_dir, f'{day}.npy'), ft)

        return pd.DataFrame([(day, i) for i in stocklist], columns=['DataDate', 'SecuCode'])

    except (FileNotFoundError, OSError) as e:
        tqdm.write(f'file/io error during {day}: {e}')
        return None
    except KeyError as e:
        tqdm.write(f'column/index missing during {day}: {e}')
        return None
    except ValueError as e:
        tqdm.write(f'value/shape error during {day}: {e}')
        return None
    except IndexError as e:
        tqdm.write(f'index out of bounds during {day}: {e}')
        return None
    except TypeError as e:
        tqdm.write(f'type mismatch during {day}: {e}')
        return None
    except MemoryError as e:
        tqdm.write(f'memory exhausted during {day}: {e}')
        return None


def load_ft_kline(
    day: str, 
    lookback: int,
    freq: Literal['one_minute', 'five_minute', 'half_hour', 'one_hour', '1d']
) -> np.ndarray:
    """
    读取已保存的Kline特征。
    """
    return np.load(
        os.path.join(TARGET_PATH, f'kline_{freq}_lag{lookback}', f'{day}.npy'),
        mmap_mode='r+'
    )



def build_Daily_kline_data_set(
    lookback: int,
    data_path: os.PathLike | str = "/data/home/kaihu/testbench/data/stock/stock_data.ftr"
) -> None:

    stock_data = pd.read_feather(data_path)
    _date = pd.to_datetime(stock_data['DataDate'])
    stock_data['DataDate'] = _date.dt.year * 10000 + _date.dt.month *100 + _date.dt.day

    ftlist = ['OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice', 'TurnoverValue', 'TurnoverVolume']

    print("正在进行全局数据透视 (Pivot)...")
    # 2. 一次性将所有数据展开 (Unstack)。
    # 这一步会将 (DataDate, SecuCode) 展开，列变成 MultiIndex(特征, 股票)
    pivoted = stock_data.set_index(["DataDate", "SecuCode"])[ftlist].unstack(level="SecuCode")
    

    # 提取维度元数据
    dates = pivoted.index.values
    stocks = pivoted.columns.levels[1].values
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
        out_dir = os.path.join(TARGET_PATH, f'kline_1d_lag{lookback}')
        _ensure_dir(out_dir)
        np.save(os.path.join(out_dir, f'{day}.npy'), ft_filtered)

        # 收集 metadata (用 list append 比在循环里建 DataFrame 快得多)
        result_idx.extend([(day, s) for s in valid_stocks])

    result = pd.DataFrame(result_idx, columns=['DataDate', 'SecuCode'])
    result.to_feather(os.path.join(TARGET_PATH, f'kline_1d_lag{lookback}.feather'))
