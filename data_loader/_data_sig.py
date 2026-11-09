import os
from typing import Literal, Optional

import pandas as pd 
import numpy as np 
from tqdm import tqdm

from .__utils__ import TARGET_PATH, _ensure_dir, _list_days_from_dir, _get_lookback_days

freq_count = {"half_hour": 8, "2h": 2, '1d': 1}

def load_sig(
    day: str, 
    freq: Literal['half_hour', '2h', '1d']
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
        当日签名数据。
    """
    data_root_dir = os.path.join("data_raw", f"sig_{freq}_raw_ordo2")

    d = pd.read_parquet(os.path.join(data_root_dir, f'{day}.parquet')) # pylint: disable=possibly-used-before-assignment

    if freq == "1d":
        time_mark = pd.DataFrame({'DataDate': int(day), 'DataTime': 1}, index=d.index)
    elif freq in ['half_hour', '2h']:
        time_mark = pd.DataFrame({'DataDate': int(day)}, index=d.index)
    else:
        raise ValueError(f"freq {freq} not supported yet, currently support half_hour, 2h, 1d")

    d = pd.concat([time_mark, d], axis=1, join='inner').reset_index(drop=False)
            
    return d.set_index(['DataDate', 'DataTime', 'SecuCode'])


def build_ft_sig(
    day: str, 
    lookback: int,
    freq: Literal['half_hour', 'half_day', '1d']
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
        data_root_dir = os.path.join("data_raw", f"sig_{freq}_raw_ordo2")
        daylist = _list_days_from_dir(data_root_dir)
        daylist_t = _get_lookback_days(day, daylist, lookback)

        if not daylist_t:
            tqdm.write(f'no available day <= {day}')
            return None

        d = pd.concat([load_sig(d_, freq) for d_ in daylist_t]).sort_index()

        # 非活跃交易填值
        d = d.ffill().dropna(how='all').astype(float)
        
        times = range(1, freq_count[freq] + 1)
        stocklist = np.unique(d.index.get_level_values("SecuCode"))
        ftlist = range(12)

        ft = np.empty((len(stocklist), lookback * len(times), len(ftlist)))
        for i, ftname in enumerate(ftlist):
            dt = d[ftname].unstack(level=[0,1]).fillna(0)
            dt = dt.reindex(index=stocklist)
            ft[:, :, i] = dt.values

        # 过滤条件
        cond = np.isnan(ft).sum(axis=(1, 2)) == 0
        ft = ft[cond, :, :]
        stocklist = [j for i, j in enumerate(stocklist) if cond[i]]

        if len(ft) == 0:
            tqdm.write(f'no valid stock after filtering on day {day}')
            return None

        # 时序标准化
        mask = np.ones(12, dtype=bool)
        mask[[0, 3]] = False
        ft[:,:,mask] = ft[:,:,mask] / (ft[:,:,mask].sum(axis=1, keepdims=True) + 1E-12)

        ft[np.isinf(ft)] = np.nan
        cond = np.isnan(ft).sum(axis=(1, 2)) == 0
        ft = ft[cond, :, :]
        stocklist = [j for i, j in enumerate(stocklist) if cond[i]]

        if len(ft) < 10:
            tqdm.write(f'too little data left on day {day}')
            return None

        out_dir = os.path.join(TARGET_PATH, f'sig_{freq}_lag{lookback}')
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


def load_ft_sig(
    day: str, 
    lookback: int,
    freq: Literal['half_hour', 'half_day', '1d']
) -> np.ndarray:
    """读取已保存的sig特征"""
    return np.load(
        os.path.join(TARGET_PATH, f'sig_{freq}_lag{lookback}', f'{day}.npy'),
        mmap_mode='r+'
    )


def build_ts_sig(
    day: str, 
    lookback: int,
    data_mark: str
) -> Optional[pd.DataFrame]:
    try:
        data_root_dir = os.path.join("data_raw", f"sig_{data_mark}")
        daylist = _list_days_from_dir(data_root_dir)
        daylist_t = _get_lookback_days(day, daylist, lookback)

        if not daylist_t:
            tqdm.write(f'no available day <= {day}')
            return None

        d = pd.concat({i: pd.read_parquet(os.path.join(data_root_dir, f"{day}.parquet")) for i, d_ in enumerate(daylist_t)})

        # 非活跃交易填值
        d = d.swaplevel().sort_index().dropna(how='all').astype(float)

        stocklist = np.unique(d.index.get_level_values("SecuCode"))
        ftlist = d.columns

        ft = np.empty((len(stocklist), lookback, len(ftlist)))
        for i, ftname in enumerate(ftlist):
            dt = d[ftname].unstack().ffill()
            dt = dt.reindex(index=stocklist)
            ft[:, :, i] = dt.values

        # 过滤条件
        cond = np.isnan(ft).sum(axis=(1, 2)) == 0
        ft = ft[cond, :, :]
        stocklist = [j for i, j in enumerate(stocklist) if cond[i]]

        if len(ft) == 0:
            tqdm.write(f'no valid stock after filtering on day {day}')
            return None

        out_dir = os.path.join(TARGET_PATH, f'sig_{data_mark}_lag{lookback}')
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