import faulthandler
faulthandler.enable()

import os
import json
# import shutil
# Limit internal C/C++ thread pools to 1 per worker
os.environ["JOBLIB_START_METHOD"] = "spawn"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import traceback
import argparse

import numpy as np
import pandas as pd
import torch
import duckdb
import signatory
from joblib import Parallel, delayed

__START_TIME__, __END_TIME__ = 93000000, 150000000
__BUDGET__ = 2_000_000
MAX_RETRY = 3
STATE_FILE = "build_sig_progress.json"   # 记录进度和失败次数


def n_sig_features(channels: int, depth: int) -> int:
    return sum(channels ** k for k in range(1, depth + 1))


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding="utf-8") as f:
            return json.load(f)
    return {"done": [], "failed": {}}


def save_state(state):
    with open(STATE_FILE, 'w', encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def is_done(file_name, depth):
    """检查最终输出文件是否存在且非空"""
    path = os.path.join("data_raw", f"sig_1d_V3_2_ordo{depth}", file_name)
    return os.path.exists(path) and os.path.getsize(path) > 0


def process_file(file_name, depth):
    db = duckdb.connect()
    db.execute("PRAGMA threads=1")
    try:
        path = f"/data/cephfs/transaction/{file_name}"

        # ---- 原子写入路径 ----
        out_dir = os.path.join("data_raw", f"sig_1d_V3_2_ordo{depth}")
        tmp_path = os.path.join(out_dir, f".tmp.{file_name}")
        final_path = os.path.join(out_dir, file_name)

        # 如果临时文件残留，先清理
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        db.execute(f'''
        CREATE TEMP TABLE temp_flagged AS
        WITH raw AS (
            SELECT 
                security_code,
                transaction_time AS DataTime,
                transaction_price AS Price,
                transaction_volume AS Volume,
                ask_appl_seq_num,
                bid_appl_seq_num,
                transaction_price * transaction_volume AS Amount
            FROM read_parquet('{path}')
            WHERE execution_type = 1
              AND transaction_price IS NOT NULL
              AND transaction_volume IS NOT NULL
              AND transaction_volume > 0
        ),
        ask_order AS (
            SELECT security_code, ask_appl_seq_num, SUM(Amount) AS order_amount
            FROM raw GROUP BY security_code, ask_appl_seq_num
            HAVING SUM(Amount) > 0
        ),
        bid_order AS (
            SELECT security_code, bid_appl_seq_num, SUM(Amount) AS order_amount
            FROM raw GROUP BY security_code, bid_appl_seq_num
            HAVING SUM(Amount) > 0
        ),
        ask_q AS (
            SELECT security_code,
                   quantile_cont(order_amount, 0.2) AS q20,
                   quantile_cont(order_amount, 0.8) AS q80
            FROM ask_order GROUP BY security_code
        ),
        bid_q AS (
            SELECT security_code,
                   quantile_cont(order_amount, 0.2) AS q20,
                   quantile_cont(order_amount, 0.8) AS q80
            FROM bid_order GROUP BY security_code
        ),
        ask_flagged AS (
            SELECT o.security_code, o.ask_appl_seq_num,
                   CASE WHEN o.order_amount <= q.q20 THEN 'small'
                        WHEN o.order_amount >= q.q80 THEN 'big'
                        ELSE 'mid' END AS ask_flag
            FROM ask_order o JOIN ask_q q ON o.security_code = q.security_code
        ),
        bid_flagged AS (
            SELECT o.security_code, o.bid_appl_seq_num,
                   CASE WHEN o.order_amount <= q.q20 THEN 'small'
                        WHEN o.order_amount >= q.q80 THEN 'big'
                        ELSE 'mid' END AS bid_flag
            FROM bid_order o JOIN bid_q q ON o.security_code = q.security_code
        )
        SELECT 
            r.security_code,
            r.DataTime,
            COALESCE(AVG(r.Price), 0) AS Price,
            COALESCE(SUM(r.Volume), 0) AS Volume,
            COALESCE(SUM(CASE WHEN af.ask_flag = 'big'  THEN r.Volume ELSE 0 END), 0) AS BigAskVolume,
            COALESCE(SUM(CASE WHEN af.ask_flag = 'small' THEN r.Volume ELSE 0 END), 0) AS SmallAskVolume,
            COALESCE(SUM(CASE WHEN bf.bid_flag = 'big'  THEN r.Volume ELSE 0 END), 0) AS BigBidVolume,
            COALESCE(SUM(CASE WHEN bf.bid_flag = 'small' THEN r.Volume ELSE 0 END), 0) AS SmallBidVolume
        FROM raw r
        LEFT JOIN ask_flagged af 
            ON r.security_code = af.security_code AND r.ask_appl_seq_num = af.ask_appl_seq_num
        LEFT JOIN bid_flagged bf 
            ON r.security_code = bf.security_code AND r.bid_appl_seq_num = bf.bid_appl_seq_num
        GROUP BY r.security_code, r.DataTime
        ORDER BY r.security_code, r.DataTime
        ''')

        df = db.query("SELECT * FROM temp_flagged").fetchdf()
        db.execute("DROP TABLE IF EXISTS temp_flagged")

        arr = {k: df[k].to_numpy().copy() for k in df.columns}
        codes = arr['security_code']

        if len(codes) == 0:
            return file_name, True, None   # 空文件算成功

        head = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
        counts = np.diff(np.r_[head, len(codes)])
        gid = np.repeat(np.arange(len(head)), counts)

        # 交易量累计
        volume = arr['Volume'].astype(np.float64)
        C = np.cumsum(volume)
        off = np.zeros(len(head)); off[1:] = C[head[1:] - 1]
        cum_vol = C - off[gid]
        log_cum = np.log1p(cum_vol + 1)
        tail = np.r_[head[1:] - 1, len(codes) - 1]
        vol_norm = log_cum / log_cum[tail][gid]

        # 价格中性
        price = arr['Price'].astype(np.float64)
        price_norm = (price + 1) / (price[head][gid] + 1)

        # 大小买卖单累计

        big_ask_vol = arr['BigAskVolume'].astype(np.float64)
        C_ba = np.cumsum(big_ask_vol)
        off_ba = np.zeros(len(head)); off_ba[1:] = C_ba[head[1:] - 1]
        cum_ba = C_ba - off_ba[gid]
        log_ba = np.log1p(cum_ba + 1)
        big_ask_norm = log_ba / log_ba[tail][gid]
        big_ask_norm = np.nan_to_num(big_ask_norm, nan=0.0, posinf=0.0, neginf=0.0)

        small_ask_vol = arr['SmallAskVolume'].astype(np.float64)
        C_sa = np.cumsum(small_ask_vol)
        off_sa = np.zeros(len(head)); off_sa[1:] = C_sa[head[1:] - 1]
        cum_sa = C_sa - off_sa[gid]
        log_sa = np.log1p(cum_sa + 1)
        small_ask_norm = log_sa / log_sa[tail][gid]
        small_ask_norm = np.nan_to_num(small_ask_norm, nan=0.0, posinf=0.0, neginf=0.0)

        big_bid_vol = arr['BigBidVolume'].astype(np.float64)
        C_bb = np.cumsum(big_bid_vol)
        off_bb = np.zeros(len(head)); off_bb[1:] = C_bb[head[1:] - 1]
        cum_bb = C_bb - off_bb[gid]
        log_bb = np.log1p(cum_bb + 1)
        big_bid_norm = log_bb / log_bb[tail][gid]
        big_bid_norm = np.nan_to_num(big_bid_norm, nan=0.0, posinf=0.0, neginf=0.0)

        small_bid_vol = arr['SmallBidVolume'].astype(np.float64)
        C_sb = np.cumsum(small_bid_vol)
        off_sb = np.zeros(len(head)); off_sb[1:] = C_sb[head[1:] - 1]
        cum_sb = C_sb - off_sb[gid]
        log_sb = np.log1p(cum_sb + 1)
        small_bid_norm = log_sb / log_sb[tail][gid]
        small_bid_norm = np.nan_to_num(small_bid_norm, nan=0.0, posinf=0.0, neginf=0.0)

        vals = np.column_stack([
            arr['DataTime'].astype(np.float64),
            price_norm,
            vol_norm,
            big_ask_norm,
            small_ask_norm,
            big_bid_norm,
            small_bid_norm,
        ])
        # 时间码中性化
        vals[:, 0] = (vals[:, 0] - __START_TIME__) / (__END_TIME__ - __START_TIME__)

        keep = counts >= 2
        kept = np.flatnonzero(keep)
        kept = kept[np.argsort(counts[kept], kind="stable")]

        n_feat = n_sig_features(vals.shape[1], depth)
        sig = np.empty((len(kept), n_feat), dtype=np.float64)

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

        # ---- 原子写入：先写 tmp，再 rename ----
        result.to_parquet(tmp_path)
        os.replace(tmp_path, final_path)

        return file_name, True, None

    except Exception as e: # pylint: disable=W0612,W0718
        tb = traceback.format_exc()
        return file_name, False, tb

    finally:
        db.close()
        # 清理可能残留的 tmp
        tmp_path = os.path.join("data_raw", f"sig_1d_V3_2_ordo{depth}", f".tmp.{file_name}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _process_file(file_name, depth):
    return process_file(file_name, depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--depth', type=int, default=2)
    parser.add_argument('-j', '--jobs', type=int, default=16)
    args = parser.parse_args()

    _depth = args.depth
    n_jobs = args.jobs

    _out_dir = os.path.join("data_raw", f"sig_1d_V3_2_ordo{_depth}")
    os.makedirs(_out_dir, exist_ok=True)

    file_names = [f for f in os.listdir("/data/cephfs/transaction/")
                  if not f.startswith("2010") and f != "file_list.txt"]
    file_names = sorted(file_names)

    _state = load_state()
    total_files = len(file_names)

    # =====================================================================
    # 外层重试循环：每轮只处理未完成的文件，失败的重试最多 MAX_RETRY 次
    # =====================================================================
    round_num = 0
    while True:
        round_num += 1

        # 1) 过滤：已完成 或 失败次数已达上限 的文件跳过
        pending = []
        for _f in file_names:
            if is_done(_f, _depth):
                if _f not in _state["done"]:
                    _state["done"].append(_f)
                continue
            fail_cnt = _state["failed"].get(_f, 0)
            if fail_cnt >= MAX_RETRY:
                print(f"[SKIP] {_f} 已失败 {fail_cnt} 次，达到上限")
                continue
            pending.append(_f)

        save_state(_state)

        if not pending:
            print(f"\n全部完成！共 {total_files} 个文件")
            break

        print(f"\n========== 第 {round_num} 轮 ==========")
        print(f"待处理: {len(pending)}/{total_files} | 已完成: {len(_state['done'])} | 失败: {len(_state['failed'])}")

        # 2) 并行处理 pending 文件（失败不抛异常，返回状态）
        results = Parallel(n_jobs=n_jobs, verbose=5, backend='multiprocessing')(
            delayed(_process_file)(f, _depth) for f in pending
        )

        # 3) 收集结果，更新状态
        new_done = 0
        new_fail = 0
        for fname, success, err in results:
            if success:
                _state["done"].append(fname)
                if fname in _state["failed"]:
                    del _state["failed"][fname]
                new_done += 1
            else:
                _state["failed"][fname] = _state["failed"].get(fname, 0) + 1
                new_fail += 1
                # 打印最后一条错误摘要（避免刷屏）
                print(f"[FAIL] {fname} 第 {_state['failed'][fname]} 次失败")
                # 把完整错误写到单独日志
                log_path = os.path.join(_out_dir, f".error.{fname}.log")
                with open(log_path, 'w', encoding="utf-8") as _f:
                    _f.write(err)

        save_state(_state)
        print(f"本轮: 成功 {new_done} | 失败 {new_fail}")

    # 清理状态文件（可选）
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)