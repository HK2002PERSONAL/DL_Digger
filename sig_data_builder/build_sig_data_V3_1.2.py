import os
import traceback
import argparse

import numpy as np
import pandas as pd
import torch
import duckdb
import signatory
from joblib import Parallel, delayed

__START_TIME__, __END_TIME__ = 93000000, 150000000
__BUDGET__ = 2_000_000           # 单 batch 的 股票数×路径长度 上限


def n_sig_features(channels: int, depth: int) -> int:
    """signatory 输出特征数 = c + c^2 + ... + c^depth"""
    return sum(channels ** k for k in range(1, depth + 1))


def process_file(file_name, depth):
    db = duckdb.connect()
    try:
        path = f"/data/cephfs/transaction/{file_name}"

        # ---- 修复6+7：纯 SQL 完成 分位数计算 + 小单/大单标记 + 时间聚合 ----
        # 不再把逐笔数据拉回 Python，内存直接减半；也不再 Python 循环算 percentile
        query = f'''
        WITH raw AS (
            SELECT 
                security_code,
                transaction_time AS DataTime,
                transaction_price AS Price,
                transaction_volume AS Volume,
                transaction_price * transaction_volume AS Amount
            FROM read_parquet('{path}')
            WHERE execution_type = 1
        ),
        stats AS (
            SELECT 
                security_code,
                quantile_cont(Amount, 0.4) AS q40,
                quantile_cont(Amount, 0.6) AS q60
            FROM raw
            GROUP BY security_code
        )
        SELECT 
            r.security_code,
            r.DataTime,
            AVG(r.Price) AS Price,
            SUM(r.Volume) AS Volume,
            SUM(CASE WHEN r.Amount <= s.q40 THEN r.Volume ELSE 0 END) AS SmallVolume,
            SUM(CASE WHEN r.Amount >= s.q60 THEN r.Volume ELSE 0 END) AS BigVolume
        FROM raw r
        JOIN stats s ON r.security_code = s.security_code
        GROUP BY r.security_code, r.DataTime
        ORDER BY r.security_code, r.DataTime
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
        cum_vol = C - off[gid]
        log_cum = cum_vol ### 
        tail = np.r_[head[1:] - 1, len(codes) - 1]
        vol_norm = log_cum / log_cum[tail][gid]

        # ---- Price 变换：除以当日首价 ----
        price = arr['Price'].astype(np.float64)
        price_norm = (price + 1) / (price[head][gid] + 1)

        # ---- 小单累计成交量 → log1p → 归一化 ----
        small_volume = arr['SmallVolume'].astype(np.float64)
        C_small = np.cumsum(small_volume)
        off_small = np.zeros(len(head)); off_small[1:] = C_small[head[1:] - 1]
        cum_small_vol = C_small - off_small[gid]
        log_cum_small = cum_small_vol ###
        small_vol_norm = log_cum_small / log_cum_small[tail][gid]
        # 修复1：同时掐掉 nan 和 inf（除零会产生 inf，不是 nan）
        small_vol_norm = np.nan_to_num(small_vol_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- 大单累计成交量 → log1p → 归一化 ----
        big_volume = arr['BigVolume'].astype(np.float64)
        C_big = np.cumsum(big_volume)
        off_big = np.zeros(len(head)); off_big[1:] = C_big[head[1:] - 1]
        cum_big_vol = C_big - off_big[gid]
        log_cum_big = cum_big_vol ### 
        big_vol_norm = log_cum_big / log_cum_big[tail][gid]
        # 修复1：同时掐掉 nan 和 inf
        big_vol_norm = np.nan_to_num(big_vol_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # 5 个通道: DataTime, price_norm, vol_norm, small_vol_norm, big_vol_norm
        vals = np.column_stack([
            arr['DataTime'].astype(np.float64),
            price_norm,
            vol_norm,
            small_vol_norm,
            big_vol_norm,
        ])
        vals[:, 0] = (vals[:, 0] - __START_TIME__) / (__END_TIME__ - __START_TIME__)

        keep = counts >= 2

        kept = np.flatnonzero(keep)
        kept = kept[np.argsort(counts[kept], kind="stable")]

        n_feat = n_sig_features(vals.shape[1], depth)
        sig = np.empty((len(kept), n_feat), dtype=np.float64)

        # ---- 贪心切块 + 零增量 padding + 批量 signatory ----
        start = 0
        while start < len(kept):
            end = start + 1
            while (end < len(kept)
                   and counts[kept[end]] * (end - start + 1) <= __BUDGET__):
                end += 1
            chunk = kept[start:end]
            Lmax = counts[chunk].max()

            batch = np.zeros((len(chunk), Lmax, vals.shape[1]), dtype=np.float64)
            for r, i in enumerate(chunk):
                L, h = counts[i], head[i]
                batch[r, :L] = vals[h:h + L]
                batch[r, L:] = vals[h + L - 1]

            sig[start:end] = signatory.signature(
                torch.from_numpy(batch), depth=depth).numpy()
            start = end

        result = pd.DataFrame(sig, index=codes[head][kept])
        result.index.name = "SecuCode"
        result = result.sort_index()
        return result.to_parquet(os.path.join("data_raw", f"sig_1d_raw_ordo{depth}_v3_1.2", file_name))

    # 修复5：确保 DuckDB 连接关闭，防止 loky 进程复用时句柄泄漏
    finally:
        db.close()


def _process_file(file_name, depth):
    try:
        return process_file(file_name, depth)
    except Exception as e:
        # 修复4：保留完整 traceback，不再吞掉原始错误堆栈
        tb = traceback.format_exc()
        raise RuntimeError(f"!!! CRASH in {file_name} !!!\n{tb}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='构建路径签名特征 V3.1 (含小单大单)')
    parser.add_argument('-d', '--depth', type=int, default=2, help='签名深度 (默认: 2)')
    parser.add_argument('-j', '--jobs', type=int, default=16, help='并行进程数 (默认: 16)')
    args = parser.parse_args()

    _depth = args.depth
    n_jobs = args.jobs

    print(f"sig_1d_raw_ordo{_depth}_v3_1.2, n_jobs={n_jobs}")

    target_path = os.path.join("data_raw", f"sig_1d_raw_ordo{_depth}_v3_1.2")

    os.makedirs(target_path, exist_ok=True)

    file_names = [f for f in os.listdir("/data/cephfs/transaction/")
                  if not f.startswith("2010") and f != "file_list.txt"]

    loop_counter = 0

    while True:

        loop_counter += 1

        done_file_names = os.listdir(target_path)
    
        todo_file_names = sorted(list(set(file_names) - set(done_file_names)))

        if len(todo_file_names) == 0:
            print(f"{len(done_file_names)} / {len(file_names)} done")
            break

        else:
            print(f"{len(todo_file_names)} / {len(file_names)} to do")

        try:
            Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(_process_file)(f, _depth) for f in todo_file_names)

        except Exception as e: # pylint:disable=W0718
            print(f"loop {loop_counter} fail due to {e}")