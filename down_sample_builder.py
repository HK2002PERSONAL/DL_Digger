# -*- coding: utf-8 -*-
"""
高频一分钟统计数据 -> 23 个日频因子 完整计算脚本
=================================================
数据表: /data/cephfs/minute/one_minute_stat/{yyyymmdd}.parquet
粒度:   trading_day x security_code x start_time (A股一分钟连续竞价)

因子清单(23 个, 6 族)
---------------------
族1 高频收益分布(minute_return):
    rv            已实现波动率        sqrt(sum r^2)                    预期负向
    rskew         高频偏度            sqrt(N)*sum r^3 / RV^3           预期负向
    rkurt         高频峰度            N*sum r^4 / RV^4                 预期负向
    rsj           上下行波动不对称    (RV+ - RV-)/RV                   预期负向
族2 量价相关与聪明钱:
    cpv           高频量价相关性      spearman corr(r, amount)         预期负向
    smart_q       聪明钱因子          VWAP_smart / VWAP_all            预期负向
族3 主动买卖订单流:
    oib           主买净额占比        sum(v_bid - v_ask)/sum(v)        短期负向
    buy_willing   买入意愿占比        sum(n_bid)/sum(n_bid + n_ask)    负向
    oib_vol       OIB 日内波动        std(分钟OIB)                     负向
    tail_amt_pct  尾盘成交占比        amount(>=14:30)/sum(amount)      负向
族4 大小单结构:
    big_net       大单净买入占比      sum(bigV_bid - bigV_ask)/sum(v)  正向
    small_net     小单净买入占比      sum(smallV_bid - smallV_ask)/v   负向
    avg_ticket    平均单笔成交额      sum(amount)/sum(count)           正向
族5 委托与撤单:
    cancel_rate   撤单率              sum(withdraw)/sum(order_vol)     负向
    cancel_asym   买卖撤单不对称      (w_bid - w_ask)/(w_bid + w_ask)  负向
    order_imb     委托不平衡          sum(ov_bid - ov_ask)/sum(ov)     短期负向
    big_order_pct 大单委托占比        sum(big_order_value)/sum(ov)     正向
    trade_density 成交密度            sum(turnover_count)/sum(order_n) 正向
族6 日内结构与行业相对:
    tail_mom      尾盘动量            sum(r, >=14:30)                  正向
    open_mom      早盘动量            sum(r, <=10:00)                  正向
    apm           上午/下午行为差异   APM 统计量(滚动20日回归残差)      看方向
    idio_vol      特质波动            std(r - indus_ret)               负向
    ind_corr      行业联动度          corr(r, indus_ret)               负向

用法
----
python 高频因子_完整计算脚本.py \
    --data-dir /data/cephfs/minute/one_minute_stat \
    --start 20150105 --end 20261231 \
    --out factors_daily.parquet \
    --workers 8 --eval

依赖: duckdb, pandas, numpy (scipy 可选)
"""

import argparse
import os
import warnings

import duckdb
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- 常量
EPS = 1e-12
OPEN_MOM_END = 600      # 10:00 之前算早盘
AM_END = 690            # 11:30 上午收盘
PM_START = 780          # 13:00 下午开盘
TAIL_START = 870        # 14:30 之后算尾盘
SMART_PCT = 0.20        # 聪明钱累计成交量阈值

# 大单/小单字段(按贵司数据字典, q1+q2 两档相加; 如需并入 _2 口径可扩展)
BIG_TRADE_BID = ["big_trade_value_bid_q1", "big_trade_value_bid_q2"]
BIG_TRADE_ASK = ["big_trade_value_ask_q1", "big_trade_value_ask_q2"]
SMALL_TRADE_BID = ["small_trade_value_bid_q1", "small_trade_value_bid_q2"]
SMALL_TRADE_ASK = ["small_trade_value_ask_q1", "small_trade_value_ask_q2"]
BIG_ORDER_BID = ["big_order_value_bid_q1", "big_order_value_bid_q2"]
BIG_ORDER_ASK = ["big_order_value_ask_q1", "big_order_value_ask_q2"]

FAM1 = ["rv", "rskew", "rkurt", "rsj"]
FAM2 = ["cpv", "smart_q"]
FAM3 = ["oib", "buy_willing", "oib_vol", "tail_amt_pct"]
FAM4 = ["big_net", "small_net", "avg_ticket"]
FAM5 = ["cancel_rate", "cancel_asym", "order_imb", "big_order_pct", "trade_density"]
FAM6 = ["tail_mom", "open_mom", "idio_vol", "ind_corr"]   # apm 在平滑阶段生成
DAILY_FACTORS = FAM1 + FAM2 + FAM3 + FAM4 + FAM5 + FAM6   # 22 个
ALL_FACTORS = DAILY_FACTORS[:18] + ["apm"] + DAILY_FACTORS[18:]  # 23 个


# ---------------------------------------------------------------- 工具
def time_to_minute(t) -> float:
    """把 start_time 整数 (HHMM, 如 930=9:30, 1456=14:56) 转为当日分钟数。"""
    t = int(t)
    # 数据格式为 HHMM: 930 -> 9*60+30=570, 1456 -> 14*60+56=896
    h, m = t // 100, t % 100
    return h * 60 + m


def load_day(data_dir: str, date: str, codes=None) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{date}.parquet")
    q = f"select * from read_parquet('{path}')"
    if codes:
        q += " where security_code in (" + ",".join(f"'{c}'" for c in codes) + ")"
    return duckdb.query(q).df()


# ---------------------------------------------------------------- 单日单票因子
def stock_daily_factors(g: pd.DataFrame) -> pd.Series:
    """g: 单只证券单日的分钟数据。返回 22 个日频原始因子 + APM 辅助列 + 收盘价。"""
    g = g.sort_values("start_time")
    r = g["minute_return"].astype(float).values
    amt = g["turnover_value"].astype(float).values
    vol = g["turnover_volume"].astype(float).replace(0, np.nan).values
    cnt = g["turnover_count"].astype(float).values
    mins = np.array([time_to_minute(t) for t in g["start_time"]])
    ind = g["indus_ret"].astype(float).fillna(0).values if "indus_ret" in g else np.zeros(len(g))

    r = np.nan_to_num(r, nan=0.0)
    n = len(r)
    rv2 = np.sum(r ** 2)
    total_amt = np.nansum(amt)
    out = {}

    # ---- 族1 高频收益分布 ----
    out["rv"] = np.sqrt(rv2)
    out["rskew"] = np.sqrt(n) * np.sum(r ** 3) / rv2 ** 1.5 if rv2 > EPS else np.nan
    out["rkurt"] = n * np.sum(r ** 4) / rv2 ** 2 if rv2 > EPS else np.nan
    rv_up, rv_dn = np.sum(r[r > 0] ** 2), np.sum(r[r < 0] ** 2)
    out["rsj"] = (rv_up - rv_dn) / rv2 if rv2 > EPS else np.nan

    # ---- 族2 量价相关 / 聪明钱 ----
    if np.nanstd(amt) > EPS and np.nanstd(r) > EPS:
        out["cpv"] = pd.Series(r).corr(pd.Series(amt), method="spearman")
    else:
        out["cpv"] = np.nan
    vwap_all = total_amt / np.nansum(vol) if np.nansum(vol) > 0 else np.nan
    s = np.abs(r) / np.sqrt(np.where(amt > 0, amt, np.nan))      # 分钟聪明度
    s = np.nan_to_num(s, nan=-1.0)
    order = np.argsort(-s)
    cumv = np.nancumsum(vol[order])
    k = int(np.searchsorted(cumv, SMART_PCT * cumv[-1])) + 1 if cumv[-1] > 0 else 0
    smart_amt = np.nansum(amt[order][:k])
    smart_vol = np.nansum(vol[order][:k])
    vwap_smart = smart_amt / smart_vol if smart_vol > 0 else np.nan
    out["smart_q"] = vwap_smart / vwap_all if vwap_all and vwap_all > 0 else np.nan

    # ---- 族3 主动买卖订单流 ----
    vb = g["turnover_value_bid"].astype(float).fillna(0).values
    va = g["turnover_value_ask"].astype(float).fillna(0).values
    out["oib"] = (vb - va).sum() / total_amt if total_amt > EPS else np.nan
    nb = g["appl_count_bid"].astype(float).fillna(0).values
    na = g["appl_count_ask"].astype(float).fillna(0).values
    out["buy_willing"] = nb.sum() / (nb + na).sum() if (nb + na).sum() > EPS else np.nan
    oib_min = np.where(vb + va > EPS, (vb - va) / (vb + va + EPS), 0.0)
    out["oib_vol"] = np.nanstd(oib_min)
    out["tail_amt_pct"] = np.nansum(amt[mins >= TAIL_START]) / total_amt if total_amt > EPS else np.nan

    # ---- 族4 大小单结构 ----
    def _colsum(cols):
        return sum(g[c].astype(float).fillna(0).sum() for c in cols if c in g)
    out["big_net"] = (_colsum(BIG_TRADE_BID) - _colsum(BIG_TRADE_ASK)) / total_amt if total_amt > EPS else np.nan
    out["small_net"] = (_colsum(SMALL_TRADE_BID) - _colsum(SMALL_TRADE_ASK)) / total_amt if total_amt > EPS else np.nan
    out["avg_ticket"] = total_amt / np.nansum(cnt) if np.nansum(cnt) > EPS else np.nan

    # ---- 族5 委托与撤单 ----
    order_vol = g["order_volume_total"].astype(float).fillna(0).sum()
    order_val = g["order_value_total"].astype(float).fillna(0).sum()
    order_cnt = g["order_count_total"].astype(float).fillna(0).sum()
    out["cancel_rate"] = g["withdraw_volume_total"].astype(float).fillna(0).sum() / order_vol if order_vol > EPS else np.nan
    wb = g["withdraw_volume_bid"].astype(float).fillna(0).sum()
    wa = g["withdraw_volume_ask"].astype(float).fillna(0).sum()
    out["cancel_asym"] = (wb - wa) / (wb + wa) if (wb + wa) > EPS else np.nan
    ovb = g["order_value_bid"].astype(float).fillna(0).sum()
    ova = g["order_value_ask"].astype(float).fillna(0).sum()
    out["order_imb"] = (ovb - ova) / order_val if order_val > EPS else np.nan
    out["big_order_pct"] = (_colsum(BIG_ORDER_BID) + _colsum(BIG_ORDER_ASK)) / order_val if order_val > EPS else np.nan
    out["trade_density"] = np.nansum(cnt) / order_cnt if order_cnt > EPS else np.nan

    # ---- 族6 日内结构与行业相对 ----
    out["tail_mom"] = np.sum(r[mins >= TAIL_START])
    out["open_mom"] = np.sum(r[(mins >= 570) & (mins <= OPEN_MOM_END)])
    out["idio_vol"] = np.nanstd(r - ind)
    out["ind_corr"] = np.corrcoef(r, ind)[0, 1] if np.nanstd(ind) > EPS and n > 2 else np.nan
    # APM 辅助列: 上午/下午收益与行业收益
    out["r_am"] = np.sum(r[(mins >= 570) & (mins < AM_END)])
    out["r_pm"] = np.sum(r[(mins >= PM_START) & (mins <= 900)])
    out["ind_am"] = np.sum(ind[(mins >= 570) & (mins < AM_END)])
    out["ind_pm"] = np.sum(ind[(mins >= PM_START) & (mins <= 900)])
    # 收盘价(供计算 T->T+1 远期收益)
    out["close"] = g["close_price"].astype(float).iloc[-1]
    return pd.Series(out)


def compute_one_date(args):
    data_dir, date, codes = args
    try:
        df = load_day(data_dir, date, codes)
    except Exception:
        return None                      # 非交易日或文件缺失
    if df.empty:
        return None
    res = (df.groupby(["trading_day", "security_code"], group_keys=False)
             .apply(stock_daily_factors))
    return res.reset_index()


# ---------------------------------------------------------------- 历史循环
def build_panel(data_dir, dates, codes=None, workers=1) -> pd.DataFrame:
    tasks = [(data_dir, d, codes) for d in dates]
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = [p for p in ex.map(compute_one_date, tasks) if p is not None]
    else:
        parts = [p for p in map(compute_one_date, tasks) if p is not None]
    if not parts:
        raise RuntimeError("没有读到任何交易日数据, 请检查 data-dir 与日期范围")
    panel = pd.concat(parts, ignore_index=True)
    panel["trading_day"] = pd.to_datetime(panel["trading_day"])
    return panel.sort_values(["security_code", "trading_day"]).reset_index(drop=True)


# ---------------------------------------------------------------- APM(滚动20日)
def add_apm(panel: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """APM: 行业调整后的上午/下午收益差的滚动 t 型统计量。

    注: 原实现分别对上午/下午收益做带截距的 OLS 回归取残差再求均值差, 但带截距
    OLS 的残差和恒为 0, 导致均值差恒为 0, 因子在任何数据上都退化。此处改为直接
    减去行业半日收益完成行业调整, 保留因子意图且不退化。
    """
    def _apm(sub):
        stat = np.full(len(sub), np.nan)
        r_am, r_pm = sub["r_am"].values, sub["r_pm"].values
        i_am, i_pm = sub["ind_am"].values, sub["ind_pm"].values
        d = (r_am - i_am) - (r_pm - i_pm)   # 逐日行业调整的上午-下午收益差
        for i in range(window - 1, len(sub)):
            sl = slice(i - window + 1, i + 1)
            s = np.nanstd(d[sl])
            stat[i] = np.nanmean(d[sl]) / s if s > EPS else np.nan
        return pd.Series(stat, index=sub.index)
    panel = panel.copy()
    panel["apm"] = (panel.groupby("security_code", group_keys=False)
                        .apply(_apm))
    return panel


# ---------------------------------------------------------------- 平滑
def smooth(panel: pd.DataFrame, window: int = 20, min_periods: int = 10) -> pd.DataFrame:
    panel = panel.copy()
    # 注意: apm 本身已是滚动 20 日统计量, 不做二次平滑
    panel[DAILY_FACTORS] = (panel.groupby("security_code")[DAILY_FACTORS]
                                 .transform(lambda x: x.rolling(window, min_periods=min_periods).mean()))
    return panel


# ---------------------------------------------------------------- 去极值/标准化/中性化
def winsorize_mad(s: pd.Series, n: float = 5.0) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if not np.isfinite(mad) or mad < EPS:
        return s
    lo, hi = med - n * 1.4826 * mad, med + n * 1.4826 * mad
    return s.clip(lo, hi)


def preprocess(panel: pd.DataFrame, factors=None,
               industry_map: pd.Series = None, size: pd.DataFrame = None) -> pd.DataFrame:
    """逐日: MAD 去极值 -> 标准化 -> (可选) 对行业哑变量 + log市值 回归取残差。
    industry_map: Series, index=security_code, value=行业代码
    size: DataFrame, 列=[trading_day, security_code, mktcap]
    """
    factors = factors or ALL_FACTORS
    df = panel.copy()

    def _one_day(g):
        X_exog = []
        if industry_map is not None:
            dummies = pd.get_dummies(g["security_code"].map(industry_map), prefix="ind")
            X_exog.append(dummies.astype(float))
        if size is not None:
            m = g.merge(size, on=["trading_day", "security_code"], how="left")["mktcap"]
            X_exog.append(np.log(m.clip(lower=1e8)).rename("log_size"))
        X = pd.concat(X_exog, axis=1) if X_exog else None
        for f in factors:
            x = winsorize_mad(g[f].astype(float))
            sd = x.std()
            z = (x - x.mean()) / sd if sd and sd > EPS else x * np.nan
            if X is not None and z.notna().sum() > (X.shape[1] + 2):
                Xm = np.column_stack([np.ones(len(g)), X.values])
                mask = z.notna().values
                try:
                    beta, *_ = np.linalg.lstsq(Xm[mask], z.values[mask], rcond=None)
                    z = z - (Xm @ beta - z.mean())
                except np.linalg.LinAlgError:
                    pass
            g[f] = z
        return g

    # 显式遍历分组(避免 pandas 3.x groupby.apply 丢弃分组键列的兼容性问题)
    parts = [_one_day(g) for _, g in df.groupby("trading_day")]
    res = pd.concat(parts, ignore_index=True)
    return res.sort_values(["security_code", "trading_day"]).reset_index(drop=True)


# ---------------------------------------------------------------- IC 检验
def add_forward_return(panel: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """用日内最后一根分钟收盘价拼出日收盘价, 计算 T -> T+horizon 远期收益。"""
    panel = panel.sort_values(["security_code", "trading_day"]).copy()
    panel["fwd_ret"] = (panel.groupby("security_code")["close"]
                             .transform(lambda x: x.shift(-horizon) / x - 1))
    return panel


def ic_report(panel: pd.DataFrame, factors=None, horizon: int = 1) -> pd.DataFrame:
    factors = factors or ALL_FACTORS
    df = add_forward_return(panel, horizon)
    rows = []
    for f in factors:
        ics = (df.dropna(subset=[f, "fwd_ret"])
                 .groupby("trading_day")
                 .apply(lambda g: g[f].corr(g["fwd_ret"], method="spearman")
                        if len(g) >= 30 else np.nan)
                 .dropna())
        if len(ics) < 5:
            continue
        rows.append({
            "factor": f,
            "IC_mean": ics.mean(),
            "IC_std": ics.std(),
            "ICIR": ics.mean() / ics.std() if ics.std() > EPS else np.nan,
            "IC_tstat": ics.mean() / (ics.std() / np.sqrt(len(ics))) if ics.std() > EPS else np.nan,
            "IC_win_rate": (np.sign(ics) == np.sign(ics.mean())).mean(),
            "n_days": len(ics),
        })
    rep = pd.DataFrame(rows).set_index("factor").sort_values("ICIR", key=abs, ascending=False)
    return rep


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="一分钟统计数据 -> 23 个日频因子")
    ap.add_argument("--data-dir", default="/data/cephfs/minute/one_minute_stat")
    ap.add_argument("--start", required=True, help="起始日期 yyyymmdd")
    ap.add_argument("--end", required=True, help="结束日期 yyyymmdd")
    ap.add_argument("--codes-file", default=None, help="可选: 证券代码清单 txt/csv")
    ap.add_argument("--industry-file", default=None, help="可选: 两列 csv (security_code, industry)")
    ap.add_argument("--size-file", default=None, help="可选: 三列 parquet/csv (trading_day, security_code, mktcap)")
    ap.add_argument("--smooth-window", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="factors_daily.parquet")
    ap.add_argument("--eval", action="store_true", help="输出 RankIC 检验报告")
    ap.add_argument("--horizon", type=int, default=1, help="IC 检验远期收益天数")
    args = ap.parse_args()

    dates = [d.strftime("%Y%m%d") for d in pd.bdate_range(args.start, args.end)]
    codes = None
    if args.codes_file:
        codes = pd.read_csv(args.codes_file, header=None)[0].astype(str).tolist()

    print(f"[1/5] 读取分钟数据并计算日频原始因子 ({len(dates)} 个交易日区间)...")
    panel = build_panel(args.data_dir, dates, codes, workers=args.workers)
    print(f"      共 {panel['trading_day'].nunique()} 个交易日, "
          f"{panel['security_code'].nunique()} 只证券, {len(panel)} 行")

    print("[2/5] 计算 APM 因子(滚动 20 日)...")
    panel = add_apm(panel)

    print(f"[3/5] {args.smooth_window} 日滚动平滑...")
    panel = smooth(panel, window=args.smooth_window)

    print("[4/5] 去极值 / 标准化 / 中性化...")
    ind_map = None
    if args.industry_file:
        ind_map = pd.read_csv(args.industry_file, index_col=0).iloc[:, 0].astype(str)
    size_df = None
    if args.size_file:
        size_df = (pd.read_parquet(args.size_file) if args.size_file.endswith(".parquet")
                   else pd.read_csv(args.size_file, parse_dates=["trading_day"]))
    panel = preprocess(panel, industry_map=ind_map, size=size_df)

    panel.to_parquet(args.out, index=False)
    print(f"      因子面板已保存: {args.out}")

    if args.eval:
        print(f"[5/5] RankIC 检验 (T -> T+{args.horizon}) ...")
        rep = ic_report(panel, horizon=args.horizon)
        rep_path = os.path.splitext(args.out)[0] + "_ic_report.csv"
        rep.to_csv(rep_path, encoding="utf-8-sig")
        print(rep.round(4).to_string())
        print(f"      IC 报告已保存: {rep_path}")


if __name__ == "__main__":
    main()