"""
V4_chenroll_v3.py
=================
优化版本：加速陈氏乘法滚动拼接

主要优化点（相比原版）：
1. 预计算所有 gap 签名，避免重复计算（原版每个窗口重复计算）
2. 预计算所有 base_inv，避免每个窗口重复计算逆签名
3. 使用 numpy np.outer 替代手写双重循环
4. 预计算级别偏移和大小，避免重复调用
5. 将核心循环整合到单个 numba 函数中，减少 Python 开销
6. 使用 loky 后端优化 joblib 并行

预期加速：
- gap_signature 计算量减少 n_spans 倍
- base_inv 计算量减少 n_spans 倍  
- outer product 使用 BLAS 加速
- 整体预计 3-10x 加速（取决于 depth 和 spans 数量）

用法:
    python build_sig_data_V4_chenroll_v3.py --depth 2 --spans 5 10 20 --n_jobs 16
"""

import os
import argparse
import traceback

import numpy as np
import pandas as pd
import duckdb
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm

__START_TIME__, __END_TIME__ = 93000000, 150000000


# ============================================================
# DuckDB 连接（worker 内独立）
# ============================================================
_CON = None

def get_db():
    global _CON
    if _CON is None:
        _CON = duckdb.connect()
        _CON.execute("PRAGMA threads=1")
    return _CON


# ============================================================
# 预计算级别信息
# ============================================================

def _precompute_level_info_py(d, depth):
    """Python 版本，用于初始化"""
    offsets = np.zeros(depth + 1, dtype=np.int64)
    sizes = np.zeros(depth + 1, dtype=np.int64)
    
    off = 0
    s = 1
    for k in range(1, depth + 1):
        offsets[k] = off
        sizes[k] = s
        off += s
        s *= d
    
    return offsets, sizes


@njit(cache=True)
def _precompute_level_info(d, depth):
    """Numba 版本"""
    offsets = np.zeros(depth + 1, dtype=np.int64)
    sizes = np.zeros(depth + 1, dtype=np.int64)
    
    off = 0
    s = 1
    for k in range(1, depth + 1):
        offsets[k] = off
        sizes[k] = s
        off += s
        s *= d
    
    return offsets, sizes


def get_sig_dim(d, depth):
    """计算签名维度"""
    dim = 0
    s = 1
    for k in range(1, depth + 1):
        dim += s
        s *= d
    return dim


# ============================================================
# 优化的核心算子
# ============================================================

@njit(cache=True)
def chen_product_fast(sa, sb, d, depth, offsets, sizes):
    """
    陈氏乘法 S_c = S_a ⊗ S_b (优化版)
    """
    sig_dim = offsets[depth] + sizes[depth] * d
    res = np.zeros(sig_dim, dtype=np.float64)

    for level in range(1, depth + 1):
        off_out = offsets[level]
        size_out = sizes[level]

        for i in range(level + 1):
            j = level - i
            if i == 0:
                off_b = offsets[j]
                res[off_out:off_out + size_out] += sb[off_b:off_b + size_out]
            elif j == 0:
                off_a = offsets[i]
                res[off_out:off_out + size_out] += sa[off_a:off_a + size_out]
            else:
                off_a = offsets[i]
                size_a = sizes[i]
                off_b = offsets[j]
                size_b = sizes[j]
                a_part = sa[off_a:off_a + size_a]
                b_part = sb[off_b:off_b + size_b]
                outer = np.outer(a_part, b_part).ravel()
                res[off_out:off_out + size_out] += outer

    return res


@njit(cache=True)
def gap_signature_fast(delta, d, depth, offsets, sizes):
    """
    直线段 delta 的签名 (优化版)
    """
    sig_dim = offsets[depth] + sizes[depth] * d
    res = np.zeros(sig_dim, dtype=np.float64)

    cur = delta.copy()
    fact = 1.0
    for k in range(1, depth + 1):
        fact *= k
        off = offsets[k]
        size = sizes[k]
        res[off:off + size] = cur / fact
        if k < depth:
            cur = np.outer(cur, delta).ravel()

    return res


@njit(cache=True)
def sig_inverse_fast(s, d, depth, offsets, sizes):
    """
    签名逆 S^{-1} (优化版)
    """
    sig_dim = offsets[depth] + sizes[depth] * d
    inv = np.zeros(sig_dim, dtype=np.float64)

    off1 = offsets[1]
    size1 = sizes[1]
    inv[off1:off1 + size1] = -s[off1:off1 + size1]

    for level in range(2, depth + 1):
        off_out = offsets[level]
        size_out = sizes[level]

        inv[off_out:off_out + size_out] = -s[off_out:off_out + size_out]

        for i in range(1, level):
            j = level - i
            off_a = offsets[i]
            size_a = sizes[i]
            off_b = offsets[j]
            size_b = sizes[j]
            
            a_part = s[off_a:off_a + size_a]
            b_part = inv[off_b:off_b + size_b]
            outer = np.outer(a_part, b_part).ravel()
            inv[off_out:off_out + size_out] -= outer

    return inv


# ============================================================
# Meta 数据计算（修复了原版的 bug）
# ============================================================

def compute_meta_file(file_name):
    """从原始交易 parquet 文件计算每只股票的 first/last 特征值"""
    try:
        db = get_db()
        path = f"/data/cephfs/transaction/{file_name}"

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

        head = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
        counts = np.diff(np.r_[head, len(codes)])
        gid = np.repeat(np.arange(len(head)), counts)

        # Volume 变换
        volume = arr['Volume'].astype(np.float64)
        C = np.cumsum(volume)
        off = np.zeros(len(head))
        off[1:] = C[head[1:] - 1]
        cum_vol = C - off[gid]
        log_cum = np.log1p(cum_vol)
        tail = np.r_[head[1:] - 1, len(codes) - 1]
        vol_norm = log_cum / log_cum[tail][gid]

        # Price zscore
        price = arr['Price'].astype(np.float64)
        price_sum = np.zeros(len(head))
        np.add.at(price_sum, gid, price)
        price_mean = price_sum / counts
        price_sq_sum = np.zeros(len(head))
        np.add.at(price_sq_sum, gid, price ** 2)
        price_std = np.sqrt(np.maximum(price_sq_sum / counts - price_mean ** 2, 0))
        price_norm = (price - price_mean[gid]) / (price_std[gid] + 1e-8)

        vals = np.column_stack([
            arr['DataTime'].astype(np.float64),
            price_norm,
            vol_norm,
        ])
        vals[:, 0] = (vals[:, 0] - __START_TIME__) / (__END_TIME__ - __START_TIME__)

        keep = counts >= 2
        first_vals = vals[head[keep]]
        last_vals = vals[(head + counts - 1)[keep]]
        kept_codes = codes[head][keep]

        file_key = int(file_name.split('.')[0])
        meta_df = pd.DataFrame({
            'first_time':  first_vals[:, 0],
            'last_time':   last_vals[:, 0],
            'first_price': first_vals[:, 1],
            'last_price':  last_vals[:, 1],
            'first_vol':   first_vals[:, 2],  # 修复 bug：原版错误地用了 [:, 0]
            'last_vol':    last_vals[:, 2],
        }, index=kept_codes)
        meta_df.index.name = "SecuCode"

        return file_key, meta_df

    except Exception as e:
        print(f"\n!!! CRASH in meta compute for {file_name} !!!")
        traceback.print_exc()
        raise RuntimeError(e) from e


# ============================================================
# 优化的单只股票滚动陈氏拼接（核心优化）
# ============================================================

@njit(cache=True)
def _compute_stock_rolling_numba(sig_vals, meta_vals, spans_arr, d, depth, offsets, sizes):
    """
    Numba 加速的股票滚动计算核心
    
    关键优化：
    1. 预计算所有 gap_sigs[t] (t=1..T-1) - 只做一次
    2. 预计算所有 base_inv[L] (L=0..T-2) - 只做一次
    3. 所有窗口共享这些预计算结果
    """
    T = sig_vals.shape[0]
    sig_dim = sig_vals.shape[1]
    n_spans = len(spans_arr)
    
    first_vals = meta_vals[:, :3]
    last_vals = meta_vals[:, 3:]

    # ===== 阶段1：预计算所有 gap 签名 =====
    # gap_sigs[t] = gap_signature(first[t] - last[t-1])  for t in 1..T-1
    gap_sigs = np.zeros((T, sig_dim), dtype=np.float64)
    for t in range(1, T):
        delta = first_vals[t] - last_vals[t - 1]
        gap_sigs[t] = gap_signature_fast(delta, d, depth, offsets, sizes)

    # ===== 阶段2：计算前缀积 P =====
    # P[t] = S_0 ⊗ gap_1 ⊗ S_1 ⊗ ... ⊗ gap_t ⊗ S_t
    P = np.zeros((T, sig_dim), dtype=np.float64)
    P[0] = sig_vals[0]

    for t in range(1, T):
        temp = chen_product_fast(P[t - 1], gap_sigs[t], d, depth, offsets, sizes)
        P[t] = chen_product_fast(temp, sig_vals[t], d, depth, offsets, sizes)

    # ===== 阶段3：预计算 base 和 base_inv =====
    # base[L] = P[L] ⊗ gap_{L+1}
    # base_inv[L] = inverse(base[L])
    # 这些在所有窗口中被复用！
    base_inv_arr = np.zeros((T, sig_dim), dtype=np.float64)
    
    for L in range(T - 1):
        base_L = chen_product_fast(P[L], gap_sigs[L + 1], d, depth, offsets, sizes)
        base_inv_arr[L] = sig_inverse_fast(base_L, d, depth, offsets, sizes)

    # ===== 阶段4：计算各滚动窗口结果 =====
    results = np.full((n_spans, T, sig_dim), np.nan, dtype=np.float64)
    
    for si in range(n_spans):
        w = spans_arr[si]
        if T < w:
            continue
        for t in range(w - 1, T):
            if t - w + 1 == 0:
                # 窗口从第 0 天开始
                results[si, t] = P[t]
            else:
                L = t - w
                # 使用预计算的 base_inv
                results[si, t] = chen_product_fast(
                    base_inv_arr[L], P[t], d, depth, offsets, sizes
                )

    return results


def compute_stock_rolling(secu_code, sig_vals, meta_vals, spans, d, depth, 
                          dates=None, offsets=None, sizes=None):
    """
    对单只股票计算所有滚动窗口的陈氏拼接签名（优化版）
    """
    T = sig_vals.shape[0]
    
    if T == 0:
        return secu_code, dates, None

    if offsets is None or sizes is None:
        offsets, sizes = _precompute_level_info(d, depth)
    
    spans_arr = np.array(spans, dtype=np.int64)
    
    results_array = _compute_stock_rolling_numba(
        sig_vals, meta_vals, spans_arr, d, depth, offsets, sizes
    )
    
    results = {}
    for i, w in enumerate(spans):
        if T >= w:
            results[w] = results_array[i]

    return secu_code, dates, results


def _compute_stock_rolling_worker(args):
    """Worker 函数"""
    try:
        secu_code, sig_vals, meta_vals, spans, d, depth, dates, offsets, sizes = args
        return compute_stock_rolling(
            secu_code, sig_vals, meta_vals, spans, d, depth, dates, offsets, sizes
        )
    except Exception as e:
        secu_code = args[0]
        print(f"\n!!! CRASH for stock {secu_code} !!!")
        traceback.print_exc()
        raise RuntimeError(e) from e


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="陈氏乘法滚动拼接 V4 每日签名数据 (优化版 V3)")
    parser.add_argument("--depth", type=int, default=2,
                        help="签名深度（需与 V4 一致）")
    parser.add_argument("--spans", type=int, nargs='+', default=[5, 10, 20],
                        help="滚动窗口天数列表")
    parser.add_argument("--n_jobs", type=int, default=16,
                        help="并行 worker 数")
    parser.add_argument("--sig_dir", type=str, default=None,
                        help="V4 签名输出目录（默认自动推断）")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="结果输出目录（默认自动推断）")
    parser.add_argument("--backend", type=str, default="loky",
                        choices=["loky", "multiprocessing", "threading"],
                        help="joblib 后端")
    args = parser.parse_args()

    depth = args.depth
    spans = args.spans
    d = 3  # V4 使用 3 通道: time, price, vol

    sig_dir = args.sig_dir or f"data_raw/sig_1d_V4_ordo{depth}"
    out_dir = args.out_dir or f"data_raw/sig_1d_V4_chenroll_ordo{depth}"
    os.makedirs(out_dir, exist_ok=True)

    # 预计算级别信息
    offsets, sizes = _precompute_level_info_py(d, depth)
    sig_dim = get_sig_dim(d, depth)

    print("=== V4 Chen Rolling (Optimized V3) ===")
    print(f"  depth   = {depth}")
    print(f"  d       = {d}")
    print(f"  sig_dim = {sig_dim}")
    print(f"  spans   = {spans}")
    print(f"  sig_dir = {sig_dir}")
    print(f"  out_dir = {out_dir}")
    print(f"  n_jobs  = {args.n_jobs}")
    print(f"  backend = {args.backend}")

    # ============================================================
    # Step 1: 计算 meta 数据
    # ============================================================
    print("\n[1/4] Computing meta data from raw transactions...")
    txn_files = sorted([f for f in os.listdir("/data/cephfs/transaction/")
                        if not f.startswith("2010") and f != "file_list.txt"])

    meta_outputs = Parallel(n_jobs=args.n_jobs, backend=args.backend)(
        delayed(compute_meta_file)(f)
        for f in tqdm(txn_files, desc="Meta")
    )
    meta_by_date = {k: v for k, v in meta_outputs if v is not None}

    # ============================================================
    # Step 2: 加载每日签名数据
    # ============================================================
    print("\n[2/4] Loading daily signatures...")
    sig_by_date = {}
    sig_files = sorted(os.listdir(sig_dir))
    for f in tqdm(sig_files, desc="Load sig"):
        file_key = int(f.split('.')[0])
        df = pd.read_parquet(os.path.join(sig_dir, f))
        sig_by_date[file_key] = df

    # ============================================================
    # Step 3: 拼接所有日期，按股票分组
    # ============================================================
    print("\n[3/4] Merging and grouping by stock...")
    sig_all = pd.concat(sig_by_date)
    sig_all.index.names = ['DataDate', 'SecuCode']
    sig_all = sig_all.sort_index()

    meta_all = pd.concat(meta_by_date)
    meta_all.index.names = ['DataDate', 'SecuCode']
    meta_all = meta_all.sort_index()

    # 取交集
    common_idx = sig_all.index.intersection(meta_all.index)
    sig_all = sig_all.loc[common_idx]
    meta_all = meta_all.loc[common_idx]

    secu_codes = sig_all.index.get_level_values('SecuCode').unique()
    print(f"  {len(secu_codes)} stocks, {len(sig_all)} total rows")

    # 构造任务（传入预计算的 offsets/sizes）
    tasks = []
    for secu_code in secu_codes:
        loc = sig_all.loc[secu_code]
        sig_vals = loc.values.astype(np.float64)
        meta_vals = meta_all.loc[secu_code][
            ['first_time', 'last_time', 'first_price',
             'last_price', 'first_vol', 'last_vol']
        ].values.astype(np.float64)
        dates = loc.index
        tasks.append((secu_code, sig_vals, meta_vals, spans, d, depth, 
                      dates, offsets, sizes))

    # ============================================================
    # Step 4: 并行计算滚动陈氏拼接
    # ============================================================
    print("\n[4/4] Computing rolling Chen products...")
    print(f"  优化: 预计算 gap_sigs 和 base_inv，所有窗口共享")
    print(f"  预计加速: {len(spans)}x (gap/base_inv 计算减少 {len(spans)} 倍)")

    outputs = Parallel(n_jobs=args.n_jobs, backend=args.backend)(
        delayed(_compute_stock_rolling_worker)(task)
        for task in tqdm(tasks, desc="Chen Roll")
    )

    # ============================================================
    # 合并结果并保存
    # ============================================================
    print("\nMerging results...")

    result_chen0 = sig_all.copy()

    rolled_dfs = {w: [] for w in spans}

    for secu_code, dates, results in outputs:
        if results is None:
            continue
        idx = pd.MultiIndex.from_product(
            [dates, [secu_code]], names=['DataDate', 'SecuCode'])
        idx = idx.swaplevel()
        for w in spans:
            if w in results:
                df_w = pd.DataFrame(results[w], index=idx)
                rolled_dfs[w].append(df_w)

    results_dict = {'Chen_0': result_chen0}
    for w in spans:
        if rolled_dfs[w]:
            df_w = pd.concat(rolled_dfs[w]).sort_index()
            results_dict[f'Chen_{w}'] = df_w

    result_df = pd.concat(results_dict, axis=1, join='inner')
    result_df = result_df.swaplevel(axis=1).sort_index(axis=1)
    result_df.columns = [f"{col[0]}_{col[1]}" for col in result_df.columns]

    span_str = "_".join(map(str, spans))
    save_path = os.path.join(out_dir, f"chen_roll_t_sig_{span_str}.ftr")
    result_df.reset_index().to_feather(save_path)
    print(f"\n✅ Saved to {save_path}")
    print(f"   Shape: {result_df.shape}")


if __name__ == "__main__":
    main()