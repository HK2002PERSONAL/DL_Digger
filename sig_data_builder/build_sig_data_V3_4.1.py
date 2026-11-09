"""
build_sig_data_V3_4.1.py
滑动窗口拼接数据计算路径签名

思路:
  1. 将排序后的文件列表拆分成 n_jobs 段; 每个 job 负责一段连续文件。
  2. 每个 job 维护一个长度 <= window 的 queue(deque):
       - 顺序遍历自己那段的每个文件, add 一个(只读一次, 减少 IO);
       - 若 queue 超过 window 则 pop 最旧的一天;
       - 当 queue 填满 window 天时, 把窗口内 window 天的数据按股票拼接,
         做与 V3_4 一致的 normalize(基准改为窗口级), 然后批量计算路径签名。
  3. 窗口内某天该股票缺失(停牌等)直接跳过, 拼接实际出现的天。
  4. 为保证每个输出日的窗口完整, 文件分段时给每个 job 预留 window-1 个
     预热文件(边界处少量重叠读取); 重叠部分只用于预热, 不重复输出。
"""

import os
import time
import traceback
from collections import deque

import numpy as np
import pandas as pd
import torch
import duckdb
import signatory
from joblib import Parallel, delayed

# ----------------------------------------------------------------------
# 可调参数
# ----------------------------------------------------------------------
__START_TIME__, __END_TIME__ = 93000000, 150000000
__DEPTH__ = 3                    # 签名阶数, 想几阶改这里
__WINDOW__ = 5                   # 滑动窗口天数(可变)
__N_JOBS__ = 16                  # 并行进程数
__BUDGET__ = 2_000_000           # 单 batch 的 股票数×路径长度 上限
__SRC_DIR__ = "/data/cephfs/transaction/"
__TAG__ = f"sig_1d_V3_4_win{__WINDOW__}_ordo{__DEPTH__}"


def n_sig_features(channels: int, depth: int) -> int:
    """signatory 输出特征数 = c + c^2 + ... + c^depth"""
    return sum(channels ** k for k in range(1, depth + 1))


# ----------------------------------------------------------------------
# 读取单个文件 -> 按股票切组的原始数据 dict
# ----------------------------------------------------------------------
def new_db():
    """为单个 job 新建独占的 duckdb 连接(单线程, 防多进程超订)。"""
    con = duckdb.connect()
    con.execute("PRAGMA threads=1")
    return con


def load_day(con, file_name):
    """读取一天的成交数据, 返回:
    {
      'code'  : (G,)      每组的股票代码
      'head'  : (G,)      每组在全局数组中的起始位置
      'counts': (G,)      每组长度
      'time'  : (N,)      float64 原始 transaction_time
      'price' : (N,)      float32 均价
      'volume': (N,)      float32 成交量
      'amount': (N,)      float32 成交额 = price * volume
    }
    数据已按 (security_code, transaction_time) 排序。
    """
    path = os.path.join(__SRC_DIR__, file_name)
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
    arr = con.query(query).fetchnumpy()
    codes = arr['security_code']
    n = len(codes)
    if n == 0:
        return None

    # SQL 已按 security_code 排序, O(N) 切组
    head = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
    counts = np.diff(np.r_[head, n])

    price = arr['Price'].astype(np.float32)
    volume = arr['Volume'].astype(np.float32)
    return {
        'code': codes[head],
        'head': head,
        'counts': counts,
        'time': arr['DataTime'].astype(np.float64),
        'price': price,
        'volume': volume,
        'amount': price * volume,
    }


# ----------------------------------------------------------------------
# 窗口拼接 + normalize + 批量签名
# ----------------------------------------------------------------------
def compute_window_signature(queue, depth):
    """queue: deque, 内含 window 天的 load_day 结果(可能含 None 空天)。
    返回 DataFrame(index=SecuCode) 或 None(窗口内无任何有效路径)。

    normalize 与 V3_4 一致, 但基准为窗口级:
      Time   : (day_idx + 日内归一化时间) / window  -> [0,1], 跨天单调递增
      Price  : (p + 1) / (窗口内首次出现的首价 + 1)
      High/Low: 窗口内 cummax/cummin ÷ 窗口首日首价
      Volume : 窗口内 cumsum ÷ 窗口末日末值
      Amount : 窗口内 cumsum ÷ 窗口末日末值
    缺失天直接跳过, 只拼接实际出现的天。
    """
    window = len(queue)
    time_span = __END_TIME__ - __START_TIME__

    # ---- 建立 code -> [(day_idx, day_dict, group_idx), ...] ----
    code_groups = {}
    for day_idx, day in enumerate(queue):
        if day is None:
            continue
        for g, code in enumerate(day['code']):
            code_groups.setdefault(code, []).append((day_idx, day, g))

    if not code_groups:
        return None

    codes_sorted = sorted(code_groups.keys())

    # ---- 逐股票拼接窗口内各天, 构造 6 维路径 ----
    paths = []          # list of (L, 6) float64
    valid_codes = []

    for code in codes_sorted:
        entries = code_groups[code]

        # 先算总长度
        total_len = 0
        for _, day, g in entries:
            total_len += day['counts'][g]
        if total_len < 2:              # 与 V3_4 一致: 路径过短跳过
            continue

        # 拼接原始数据
        all_time = np.empty(total_len, dtype=np.float64)
        all_price = np.empty(total_len, dtype=np.float64)
        all_vol = np.empty(total_len, dtype=np.float64)
        all_amo = np.empty(total_len, dtype=np.float64)
        all_dayidx = np.empty(total_len, dtype=np.float64)

        offset = 0
        for day_idx, day, g in entries:
            s = day['head'][g]
            e = s + day['counts'][g]
            seg_len = e - s
            all_time[offset:offset + seg_len] = day['time'][s:e]
            all_price[offset:offset + seg_len] = day['price'][s:e]
            all_vol[offset:offset + seg_len] = day['volume'][s:e]
            all_amo[offset:offset + seg_len] = day['amount'][s:e]
            all_dayidx[offset:offset + seg_len] = day_idx
            offset += seg_len

        # ---- 构造 6 维路径并 normalize ----
        path = np.empty((total_len, 6), dtype=np.float64)

        # Time: (day_idx + 日内归一化时间) / window
        intraday_norm = (all_time - __START_TIME__) / time_span
        path[:, 0] = (all_dayidx + intraday_norm) / window

        # Price / High / Low: 除以窗口首日首价
        base_price = all_price[0] + 1.0
        path[:, 1] = (all_price + 1.0) / base_price
        path[:, 2] = (np.maximum.accumulate(all_price) + 1.0) / base_price
        path[:, 3] = (np.minimum.accumulate(all_price) + 1.0) / base_price

        # Volume / Amount: 窗口内 cumsum ÷ 窗口末日末值
        cum_vol = np.cumsum(all_vol)
        cum_amo = np.cumsum(all_amo)
        path[:, 4] = (cum_vol + 1.0) / (cum_vol[-1] + 1.0)
        path[:, 5] = (cum_amo + 1.0) / (cum_amo[-1] + 1.0)

        paths.append(path)
        valid_codes.append(code)

    if not paths:
        return None

    # ---- 按路径长度升序, 贪心切块 + 零增量 padding + 批量 signatory ----
    lengths = np.array([p.shape[0] for p in paths])
    order = np.argsort(lengths, kind="stable")
    n_feat = n_sig_features(paths[0].shape[1], depth)
    sig = np.empty((len(order), n_feat), dtype=np.float64)

    start = 0
    while start < len(order):
        end = start + 1
        while (end < len(order)
               and lengths[order[end]] * (end - start + 1) <= __BUDGET__):
            end += 1
        chunk = order[start:end]
        Lmax = lengths[chunk[-1]]       # 已按长度升序
        n_col = paths[chunk[0]].shape[1]

        batch = np.zeros((len(chunk), Lmax, n_col), dtype=np.float64)
        for r, i in enumerate(chunk):
            p = paths[i]
            L = p.shape[0]
            batch[r, :L] = p
            batch[r, L:] = p[-1]        # 重复末点 = 零增量, 签名不变

        sig[start:end] = signatory.signature(
            torch.from_numpy(batch), depth=depth).numpy()
        start = end

    result = pd.DataFrame(sig, index=np.array(valid_codes)[order])
    result.index.name = "SecuCode"
    return result.sort_index()


# ----------------------------------------------------------------------
# 单个 job: 顺序滑动窗口
# ----------------------------------------------------------------------
def process_job(job_id, _file_names, _out_indices, window, depth, _target_path):
    """file_names: 该 job 需要顺序读取的连续文件(含预热段)。
    out_indices: file_names 中哪些位置是"窗口右端点"需要输出。
    每个文件只读一次, queue 自动维护 pop/add。
    每个 job 独占一个 duckdb 连接, 避免跨进程共享导致性能不稳定。
    """
    con = new_db()
    queue = deque(maxlen=window)
    n_total = len(_file_names)
    n_done = 0
    t_job = time.time()
    print(f"[job {job_id}] start, {n_total} files to scan, "
          f"{len(_out_indices)} windows to output", flush=True)

    try:
        for _idx, f in enumerate(_file_names):
            t_load = time.time()
            try:
                day = load_day(con, f)
            except Exception:          # pylint: disable=W0718
                print(f"[job {job_id}] !!! load fail {f}", flush=True)
                traceback.print_exc()
                day = None
            n_rows = 0 if day is None else int(day['counts'].sum())
            print(f"[job {job_id}] load {f}: {n_rows} rows, "
                  f"{time.time() - t_load:.1f}s", flush=True)

            queue.append(day)          # add 一个; maxlen 自动 pop 最旧的

            if _idx in _out_indices and len(queue) == window:
                t_sig = time.time()
                df = compute_window_signature(queue, depth)
                if df is not None:
                    df.to_parquet(os.path.join(_target_path, f))
                    n_done += 1
                    print(f"[job {job_id}] ({n_done}/{len(_out_indices)}) "
                          f"output {f}: {len(df)} stocks, sig cost "
                          f"{time.time() - t_sig:.1f}s", flush=True)
                else:
                    print(f"[job {job_id}] skip {f}: empty window", flush=True)
    finally:
        con.close()

    print(f"[job {job_id}] done, {n_done} outputs, "
          f"total {time.time() - t_job:.1f}s", flush=True)


# ----------------------------------------------------------------------
# 文件列表拆分: 输出右端点均分 + 预热上下文
# ----------------------------------------------------------------------
def split_jobs(all_files, window, n_jobs):
    """把输出日(右端点)均分给 n_jobs 个 job, 每个 job 带 window-1 预热文件。
    返回: [(job_files, out_indices), ...]
    """
    F = len(all_files)
    n_out = F - window + 1             # 右端点索引 window-1 .. F-1
    if n_out <= 0:
        return []

    n_jobs = min(n_jobs, n_out)
    bounds = np.linspace(0, n_out, n_jobs + 1, dtype=int)
    _jobs = []
    for j in range(n_jobs):
        out_lo, out_hi = bounds[j], bounds[j + 1]      # [lo, hi)
        if out_lo >= out_hi:
            continue
        right_lo = out_lo + window - 1                 # 全局文件索引(第一个输出)
        right_hi = out_hi + window - 2                 # 全局文件索引(最后一个输出)
        file_lo = max(0, right_lo - window + 1)        # 预热上下文起点
        _job_files = all_files[file_lo: right_hi + 1]
        # 该 job 内需要输出的右端点位置(局部索引)
        _out_indices = set(range(right_lo - file_lo, right_hi - file_lo + 1))
        _jobs.append((_job_files, _out_indices))
    return _jobs


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
if __name__ == "__main__":

    print(__TAG__)

    target_path = os.path.join("data_raw", __TAG__)
    os.makedirs(target_path, exist_ok=True)

    loop_counter = 0

    while True:

        loop_counter += 1

        file_names = sorted([f for f in os.listdir(__SRC_DIR__)
                             if not f.startswith("2010") and f != "file_list.txt"])

        done_file_names = set(os.listdir(target_path))

        # 找出还没产出的右端点日(输出文件名 = 右端点文件名)
        todo_rights = set(f for i, f in enumerate(file_names)
                          if i >= __WINDOW__ - 1 and f not in done_file_names)

        if len(todo_rights) == 0:
            print(f"all done, total files {len(file_names)}")
            break

        n_out_total = max(0, len(file_names) - __WINDOW__ + 1)
        print(f"{len(todo_rights)} / {n_out_total} to do")

        # 按全部右端点切 job, 但只保留有待做输出的 job
        jobs = split_jobs(file_names, __WINDOW__, __N_JOBS__)
        run_jobs = []
        for job_files, out_indices in jobs:
            keep = set()
            for idx in out_indices:
                if job_files[idx] in todo_rights:
                    keep.add(idx)
            if keep:
                run_jobs.append((job_files, keep))

        try:
            Parallel(n_jobs=min(__N_JOBS__, len(run_jobs)), verbose=5)(
                delayed(process_job)(j, jf, oi, __WINDOW__, __DEPTH__, target_path)
                for j, (jf, oi) in enumerate(run_jobs))

        except Exception as e:         # pylint: disable=W0718
            print(f"loop {loop_counter} fail due to {e}")