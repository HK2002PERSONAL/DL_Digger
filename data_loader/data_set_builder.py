import os
from typing import Literal

import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from .__utils__ import SOURCE_PATH, TARGET_PATH, _ensure_dir, _list_days_from_dir
from ._data_kline import build_ft_kline_min, build_Daily_kline_data_set
from ._data_sig import build_ft_sig, build_ts_sig
from ._data_hff import build_Daily_hff_data_set # pylint:disable=W0611

def build_kline_data_set(
    lookback: int, 
    freq: Literal['one_minute', 'five_minute', 'fifteen_minute', 'half_hour', 'one_hour', '1d'],
    max_worker: int = 32,
    server_mode: bool = False
) -> None:
    """
    多进程构建特征集，并汇总索引文件为 feather。

    参数
    ----
    - lookback : int
        回看窗口长度。
    - freq : str
        特征频率，支持：'one_minute', 'five_minute', 'half_hour', 'fifteen_minute', 'one_hour'。
    - max_worker : int, default=32
        并行进程数。

    返回
    ----
    None
    """
    if freq == '1d':
        return build_Daily_kline_data_set(lookback=lookback)

    out_dir = os.path.join(TARGET_PATH, f'kline_{freq}_lag{lookback}')
    if os.path.exists(out_dir):
        print('data built')
        return
    else:
        _ensure_dir(out_dir)
    
    if freq in ['fifteen_minute', 'half_hour', 'one_hour']:
        daylist = sorted(_list_days_from_dir(os.path.join(SOURCE_PATH, "five_minute")))[lookback:]
    else:
        daylist = sorted(_list_days_from_dir(os.path.join(SOURCE_PATH, freq)))[lookback:]
        
    print(f"{len(daylist)} days need to be processed")

    if server_mode:
        idx = Parallel(n_jobs=max_worker, timeout=99999, verbose=5)(
            delayed(build_ft_kline_min)(day, lookback, freq)
            for day in daylist)
    else:
        idx = Parallel(n_jobs=max_worker, timeout=99999)(
            delayed(build_ft_kline_min)(day, lookback, freq)
            for day in tqdm(daylist, desc=f'Building feature {freq} lookback {lookback}')
        )

    idx_df: pd.DataFrame = pd.concat([i for i in idx if i is not None])
    idx_df.to_feather(os.path.join(TARGET_PATH, f'kline_{freq}_lag{lookback}.feather'))


def build_sig_data_set(
    lookback: int, 
    freq: Literal['half_hour', '2h', '1d'],
    max_worker: int = 32
) -> None:
    """
    多进程构建签名特征集，并汇总索引文件为 feather。

    参数
    ----
    - lookback : int
        回看窗口长度。
    - freq : str
        特征频率，支持：'half_hour', '2h', '1d'。
    - max_worker : int, default=32
        并行进程数。

    返回
    ----
    None
    """
    if freq in ['half_hour', '2h', '1d']:
        data_root_dir = os.path.join("data_raw", f"sig_{freq}_raw_ordo2")
        build_ft = build_ft_sig

    else:
        data_root_dir = os.path.join("data_raw", f"sig_{freq}")
        build_ft = build_ts_sig

    out_dir = os.path.join(TARGET_PATH, f'sig_{freq}_lag{lookback}')

    if os.path.exists(out_dir):
        print('data built')
        return
    else:
        _ensure_dir(out_dir)
    
    daylist = sorted(_list_days_from_dir(data_root_dir))[lookback:]
        
    print(f"{len(daylist)} days need to be processed")

    idx = Parallel(n_jobs=max_worker, timeout=99999)(
        delayed(build_ft)(day, lookback, freq)
        for day in tqdm(daylist, desc=f'Building sig {freq} lookback {lookback}')
    )

    idx_df: pd.DataFrame = pd.concat([i for i in idx if i is not None])
    idx_df.to_feather(os.path.join(TARGET_PATH, f'sig_{freq}_lag{lookback}.feather'))
