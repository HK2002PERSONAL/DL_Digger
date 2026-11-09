import os
import traceback

import numpy as np
import pandas as pd
import torch
import duckdb
import signatory
from joblib import Parallel, delayed

__START_TIME__, __END_TIME__ = 93000000, 150000000
__DEPTH__ = 2                    # 想几阶改这里
__BUDGET__ = 2_000_000           # 单 batch 的 股票数×路径长度 上限（2e6×3×8B ≈ 48MB）


def n_sig_features(channels: int, depth: int) -> int:
    """signatory 输出特征数 = c + c^2 + ... + c^depth"""
    return sum(channels ** k for k in range(1, depth + 1))


_CON = None

def get_db():
    global _CON  # pylint: disable=W0603
    if _CON is None:
        _CON = duckdb.connect()
        _CON.execute("PRAGMA threads=1")   # worker 内单线程，防 32 进程超订
    return _CON



def process_file(file_name):
    db = get_db()
    path = f"/data/cephfs/transaction/{file_name}"

    # ---- 唯一的 I/O：整个文件只扫一遍 ----
    query = f'''
    SELECT security_code,
           transaction_time AS DataTime,
           AVG(transaction_price)  AS Price,
           SUM(transaction_volume) AS Volume
    FROM read_parquet('{path}')
    WHERE execution_type = 1
    GROUP BY security_code, transaction_time
    ORDER BY security_code, transaction_time
    '''
    arr = db.query(query).fetchnumpy()
    codes = arr['security_code']

    # ---- SQL 已按 security_code 排序，先 O(N) 切组 ----
    head = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
    counts = np.diff(np.r_[head, len(codes)])
    gid = np.repeat(np.arange(len(head)), counts)

    # ---- Volume 变换：组内 cumsum → log1p → 除以当日末值 ----
    volume = arr['Volume'].astype(np.float64)
    C = np.cumsum(volume)
    off = np.zeros(len(head)); off[1:] = C[head[1:] - 1]
    cum_vol = C - off[gid]                         # 组内累计成交量
    log_cum = np.log1p(cum_vol)
    tail = np.r_[head[1:] - 1, len(codes) - 1]     # 每组末行位置
    vol_norm = log_cum / log_cum[tail][gid]        # 日内累计成交占比 → [0, 1]

    # ---- Price 变换：除以当日首价 ----
    price = arr['Price'].astype(np.float64)
    price_norm = (price + 1) / (price[head][gid] + 1)         # 相对价格路径，起点 = 1

    vals = np.column_stack([
        arr['DataTime'].astype(np.float64),
        price_norm,
        vol_norm,
    ])
    vals[:, 0] = (vals[:, 0] - __START_TIME__) / (__END_TIME__ - __START_TIME__)

    keep = counts >= 2                     # 等价于原来 signatory 短路径 ValueError 跳过

    kept = np.flatnonzero(keep)
    kept = kept[np.argsort(counts[kept], kind="stable")]   # 按路径长度升序

    n_feat = n_sig_features(vals.shape[1], __DEPTH__)
    sig = np.empty((len(kept), n_feat), dtype=np.float64)

    # ---- 贪心切块 + 零增量 padding + 批量 signatory ----
    start = 0
    while start < len(kept):
        end = start + 1
        while (end < len(kept)
               and counts[kept[end]] * (end - start + 1) <= __BUDGET__):
            end += 1
        chunk = kept[start:end]
        Lmax = counts[chunk].max()         # chunk 已按长度升序，即最后一只

        batch = np.zeros((len(chunk), Lmax, vals.shape[1]), dtype=np.float64)
        for r, i in enumerate(chunk):
            L, h = counts[i], head[i]
            batch[r, :L] = vals[h:h + L]
            batch[r, L:] = vals[h + L - 1]  # 重复末点 = 零增量，签名不变

        sig[start:end] = signatory.signature(
            torch.from_numpy(batch), depth=__DEPTH__).numpy()
        start = end

    result = pd.DataFrame(sig, index=codes[head][kept])
    result.index.name = "SecuCode"
    result = result.sort_index()           # 恢复按代码排序，与原版一致
    return result.to_parquet(os.path.join("data_raw", f"sig_1d_raw_ordo{__DEPTH__}", file_name))

def _process_file(file_name):
    try:
        return process_file(file_name)
    except Exception as e:
        print(f"\n!!! CRASH in {file_name} !!!")
        traceback.print_exc()
        raise RuntimeError(e) #pylint: disable=W0707

if __name__ == "__main__":

    print(f"sig_1d_raw_ordo{__DEPTH__}")

    os.makedirs(os.path.join("data_raw", f"sig_1d_raw_ordo{__DEPTH__}"), exist_ok=True)

    file_names = [f for f in os.listdir("/data/cephfs/transaction/")
                  if not f.startswith("2010") and f != "file_list.txt"]

    Parallel(n_jobs=64, verbose=5)(
        delayed(_process_file)(f) for f in file_names)