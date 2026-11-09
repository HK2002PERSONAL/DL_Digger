"""
V4_chenroll_nogap.py
====================
直接对 V4 每日签名做滚动陈氏拼接，**不使用** 日间 gap 签名 / meta 数据。

原理:
    窗口签名 S_win = S_{t-w+1} ⊗ ... ⊗ S_t = P[t-w]^{-1} ⊗ P[t]
    其中 P[t] = S_0 ⊗ S_1 ⊗ ... ⊗ S_t 为前缀积。
    签名平移不变 ⇒ 各日路径可任意平移后首尾相接，Chen 乘法合法。

只做拼接，**不做**窗口内 zscore 标准化。

加速点（相对 V4_chenroll.py）:
  1. 无 meta / 无 DuckDB / 无原始逐笔数据读取
  2. 滚动拼接整体包进单个 @njit 核，消除逐次调用开销
  3. P[L]^{-1} 按 L 预计算一次，跨窗口共享，每个 (w,t) 仅剩 1 次 chen_product
  4. numpy 边界切片构造任务（不逐股 pandas .loc）
  5. threading 并行（njit 释放 GIL，省进程间 pickle）

用法:
    python V4_chenroll_nogap.py --depth 2 --spans 5 10 20 --n_jobs 16
"""

import os
import argparse
import traceback

import numpy as np
import pandas as pd
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm


# ============================================================
# 陈氏乘法核心算子（Numba @njit，flat 数组，支持任意 depth）
# 布局与 signatory.signature 输出一致: [S^1(d), S^2(d^2), ..., S^depth]
# ============================================================

@njit(cache=True)
def _level_offset(k, d):
    """level k (1-based) 在 flat 向量中的起始 offset"""
    off = 0
    for i in range(1, k):
        p = 1
        for _ in range(i):
            p *= d
        off += p
    return off


@njit(cache=True)
def _level_size(k, d):
    """level k (1-based) 的大小 = d^k"""
    s = 1
    for _ in range(k):
        s *= d
    return s


@njit(cache=True)
def chen_product(sa, sb, d, depth):
    """陈氏乘法 S_c = S_a ⊗ S_b"""
    sig_dim = _level_offset(depth, d) + _level_size(depth, d)
    res = np.zeros(sig_dim, dtype=np.float64)

    for level in range(1, depth + 1):
        off_out = _level_offset(level, d)
        size_out = _level_size(level, d)

        for i in range(level + 1):
            j = level - i
            if i == 0:
                off_b = _level_offset(j, d)
                for idx in range(size_out):
                    res[off_out + idx] += sb[off_b + idx]
            elif j == 0:
                off_a = _level_offset(i, d)
                for idx in range(size_out):
                    res[off_out + idx] += sa[off_a + idx]
            else:
                off_a = _level_offset(i, d)
                size_a = _level_size(i, d)
                off_b = _level_offset(j, d)
                size_b = _level_size(j, d)
                for ia in range(size_a):
                    for jb in range(size_b):
                        res[off_out + ia * size_b + jb] += sa[off_a + ia] * sb[off_b + jb]

    return res


@njit(cache=True)
def sig_inverse(s, d, depth):
    """
    签名逆 S^{-1}，使得 S ⊗ S^{-1} = identity
    递推: T^1 = -S^1;  T^k = -S^k - sum_{i=1}^{k-1} S^i ⊗ T^{k-i}
    """
    sig_dim = _level_offset(depth, d) + _level_size(depth, d)
    inv = np.zeros(sig_dim, dtype=np.float64)

    off1 = _level_offset(1, d)
    size1 = _level_size(1, d)
    for idx in range(size1):
        inv[off1 + idx] = -s[off1 + idx]

    for level in range(2, depth + 1):
        off_out = _level_offset(level, d)
        size_out = _level_size(level, d)

        for idx in range(size_out):
            inv[off_out + idx] = -s[off_out + idx]

        for i in range(1, level):
            j = level - i
            off_a = _level_offset(i, d)
            size_a = _level_size(i, d)
            off_b = _level_offset(j, d)
            size_b = _level_size(j, d)
            for ia in range(size_a):
                for jb in range(size_b):
                    inv[off_out + ia * size_b + jb] -= s[off_a + ia] * inv[off_b + jb]

    return inv


# ============================================================
# 单只股票滚动拼接的 njit 核（前缀积 + 逆预计算 + 全部窗口）
# ============================================================

@njit(cache=True)
def _chen_roll_core(sig_vals, spans_arr, d, depth):
    """
    sig_vals : (T, sig_dim) 每日签名，按日期升序
    spans_arr: (nsp,) 窗口天数
    Returns  : (nsp, T, sig_dim)，T < w 的窗口整窗为 NaN，t < w-1 的行为 NaN
    """
    T, sig_dim = sig_vals.shape
    nsp = spans_arr.shape[0]

    # ---- 阶段1：前缀积 P[t] = S_0 ⊗ ... ⊗ S_t ----
    P = np.empty((T, sig_dim), dtype=np.float64)
    P[0] = sig_vals[0]
    for t in range(1, T):
        P[t] = chen_product(P[t - 1], sig_vals[t], d, depth)

    # ---- 阶段2：BI[L] = P[L]^{-1}，只算一次，跨窗口共享 ----
    BI = np.empty((T, sig_dim), dtype=np.float64)  # BI[T-1] 不使用
    for L in range(T - 1):
        BI[L] = sig_inverse(P[L], d, depth)

    # ---- 阶段3：S_win = BI[t-w] ⊗ P[t]，每个 (w,t) 仅一次 chen_product ----
    outs = np.full((nsp, T, sig_dim), np.nan, dtype=np.float64)
    for wi in range(nsp):
        w = spans_arr[wi]
        if T < w:
            continue
        for t in range(w - 1, T):
            L = t - w
            if L < 0:
                outs[wi, t] = P[t]          # 窗口从第 0 天开始
            else:
                outs[wi, t] = chen_product(BI[L], P[t], d, depth)

    return outs


def compute_stock_rolling(secu_code, sig_vals, spans, d, depth, dates=None):
    """对单只股票计算所有滚动窗口的陈氏拼接签名"""
    T = sig_vals.shape[0]
    if T == 0:
        return secu_code, dates, None

    spans_arr = np.asarray(spans, dtype=np.int64)
    outs = _chen_roll_core(sig_vals, spans_arr, d, depth)

    results = {}
    for wi, w in enumerate(spans):
        if T >= w:
            results[w] = outs[wi]
    return secu_code, dates, results


def _compute_stock_rolling(args):
    try:
        return compute_stock_rolling(*args)
    except Exception as e:
        print(f"\n!!! CRASH for stock {args[0]} !!!")
        traceback.print_exc()
        raise RuntimeError(e) from e


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="V4 每日签名滚动陈氏拼接（无 gap / 无 meta 版）")
    parser.add_argument("--depth", type=int, default=2,
                        help="签名深度（需与 V4 生成时一致）")
    parser.add_argument("--spans", type=int, nargs='+', default=[5, 10, 20],
                        help="滚动窗口天数列表")
    parser.add_argument("--n_jobs", type=int, default=16,
                        help="并行线程数")
    parser.add_argument("--sig_dir", type=str, default=None,
                        help="V4 签名输出目录（默认自动推断）")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="结果输出目录（默认自动推断）")
    args = parser.parse_args()

    depth = args.depth
    spans = sorted(args.spans)
    d = 3  # V4 使用 3 通道: time, price, vol

    sig_dir = args.sig_dir or f"data_raw/sig_1d_V3_ordo{depth}"
    out_dir = args.out_dir or f"data_raw/sig_1d_V3_chenroll_nogap_ordo{depth}"
    os.makedirs(out_dir, exist_ok=True)

    print("=== V4 Chen Rolling (no-gap) ===")
    print(f"  depth   = {depth}")
    print(f"  spans   = {spans}")
    print(f"  sig_dir = {sig_dir}")
    print(f"  out_dir = {out_dir}")
    print(f"  n_jobs  = {args.n_jobs}")

    # ============================================================
    # Step 1: 加载每日签名数据（threading 并行读 parquet）
    # ============================================================
    print("\n[1/3] Loading daily signatures...")
    sig_files = sorted(os.listdir(sig_dir))

    def _load_sig(f):
        return int(f.split('.')[0]), pd.read_parquet(os.path.join(sig_dir, f))

    loaded = Parallel(n_jobs=args.n_jobs, backend='threading')(
        delayed(_load_sig)(f) for f in tqdm(sig_files, desc="Load sig"))
    sig_by_date = dict(loaded)

    # ============================================================
    # Step 2: 拼接所有日期，numpy 切片按股票构造任务
    # ============================================================
    print("\n[2/3] Merging and grouping by stock...")
    sig_all = pd.concat(sig_by_date)
    sig_all.index.names = ['DataDate', 'SecuCode']
    sig_all = sig_all.sort_index()
    sig_all = sig_all.loc[sig_all.index.get_level_values("SecuCode") != '']
    sig_all = sig_all[~sig_all.index.duplicated(keep="first")]

    # 按 (SecuCode, DataDate) 排序，用 numpy 边界切片代替逐股 .loc
    sig_sorted = sig_all.sort_index(level=['SecuCode', 'DataDate'])
    sig_mat = np.ascontiguousarray(sig_sorted.values, dtype=np.float64)
    codes = sig_sorted.index.get_level_values('SecuCode').to_numpy()
    dates_arr = sig_sorted.index.get_level_values('DataDate').to_numpy()

    head = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, len(codes)]
    secu_codes = codes[head[:-1]]
    print(f"  {len(secu_codes)} stocks, {len(sig_all)} total rows")

    tasks = []
    for i in range(len(secu_codes)):
        s, e = head[i], head[i + 1]
        tasks.append((secu_codes[i], sig_mat[s:e], spans, d, depth, dates_arr[s:e]))

    # ============================================================
    # Step 3: 并行计算滚动陈氏拼接（threading：njit 释放 GIL，免 pickle）
    # ============================================================
    print("\n[3/3] Computing rolling Chen products...")
    outputs = Parallel(n_jobs=args.n_jobs, backend='threading')(
        delayed(_compute_stock_rolling)(task)
        for task in tqdm(tasks, desc="Chen Roll")
    )

    # ============================================================
    # 合并结果并保存
    # ============================================================
    print("\nMerging results...")

    result_chen0 = sig_all.copy()  # Chen_0 = 原始单日签名

    rolled_dfs = {w: [] for w in spans}
    for secu_code, dates, results in outputs:
        if results is None:
            continue
        idx = pd.MultiIndex.from_product(
            [dates, [secu_code]], names=['DataDate', 'SecuCode'])
        idx = idx.swaplevel()
        for w in spans:
            if w in results:
                rolled_dfs[w].append(pd.DataFrame(results[w], index=idx))

    results_dict = {'Chen_1': result_chen0}
    for w in spans:
        if rolled_dfs[w]:
            results_dict[f'Chen_{w}'] = pd.concat(rolled_dfs[w]).sort_index()
            print(results_dict[f'Chen_{w}'])

    result_df = pd.concat(results_dict, axis=1, join='inner')
    result_df = result_df.swaplevel(axis=1).sort_index(axis=1)
    result_df.columns = [f"{col[0]}_{col[1]}" for col in result_df.columns]

    span_str = "_".join(map(str, spans))
    save_path = os.path.join(out_dir, f"chen_roll_V3_sig_{span_str}.ftr")
    result_df.reset_index().to_feather(save_path)
    print(f"\n✅ Saved to {save_path}")
    print(f"   Shape: {result_df.shape}")


if __name__ == "__main__":
    main()