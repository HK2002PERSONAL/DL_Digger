# -*- coding: utf-8 -*-
"""
生成日频因子面板 DataFrame
================================
输出格式: pd.DataFrame (仅支持 parquet)
    index : MultiIndex [DataDate (datetime), SecuCode (str)]
    columns: 23 个因子 (feature)

处理流程: build_panel -> add_apm -> preprocess(去极值+标准化) -> 输出
(不做平滑)

用法
----
python build_factor_panel.py \
    --out factors_panel.parquet \
    [--workers 8] \
    [--industry-file ind.csv] [--size-file size.parquet]

说明: 保留全部证券, DataDate 固定为 datetime 格式。

依赖: down_sample_builder.py (同目录)
"""

import argparse
import os

import pandas as pd

from down_sample_builder import (
    build_panel,
    add_apm,
    preprocess,
    ALL_FACTORS,
)


# ---------------------------------------------------------------- 日期
MIN_DATE = "20150504"   # 订单流字段自 20150504 起才有有效值, 之前全为 0 导致因子 NaN


def scan_dates(data_dir: str) -> list:
    """扫描数据目录下全部 {yyyymmdd}.parquet, 仅保留 >= MIN_DATE 的日期。"""
    dates = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(data_dir)
        if f.endswith(".parquet") and len(os.path.splitext(f)[0]) == 8
        and os.path.splitext(f)[0].isdigit()
    )
    if not dates:
        raise RuntimeError(f"{data_dir} 下未找到任何 {{yyyymmdd}}.parquet 数据文件")
    dates = [d for d in dates if d >= MIN_DATE]
    if not dates:
        raise RuntimeError(f"过滤 (>= {MIN_DATE}) 后无可用交易日")
    return dates


# ---------------------------------------------------------------- 校验
def _validate(panel: pd.DataFrame, expected_factors) -> None:
    """对最终面板做严谨性校验, 任何异常直接抛出。"""
    # 1) 因子列完整性
    missing = [f for f in expected_factors if f not in panel.columns]
    if missing:
        raise ValueError(f"缺少因子列: {missing}")

    # 2) 索引唯一性 (DataDate, SecuCode)
    idx = panel.index
    if idx.duplicated().any():
        n_dup = int(idx.duplicated().sum())
        raise ValueError(f"索引 (DataDate, SecuCode) 存在 {n_dup} 条重复记录")

    # 3) 行数与交易日/证券数一致性
    n_days = panel.index.get_level_values("DataDate").nunique()
    n_codes = panel.index.get_level_values("SecuCode").nunique()
    if len(panel) == 0:
        raise RuntimeError("输出面板为空")
    print(f"  [校验] 交易日数={n_days}, 证券数={n_codes}, 总行数={len(panel)}")

    # 4) 每只证券的交易日覆盖(检测严重缺失)
    per_code = panel.groupby(level="SecuCode").size()
    if per_code.min() == 0:
        raise RuntimeError("存在无数据的证券")
    print(f"  [校验] 单证券行数: min={per_code.min()}, "
          f"median={int(per_code.median())}, max={per_code.max()}")

    # 5) 因子 NaN 覆盖率
    nan_ratio = panel[expected_factors].isna().mean()
    print("  [校验] 因子 NaN 覆盖率 (%):")
    for f, r in nan_ratio.sort_values(ascending=False).items():
        flag = "  <-- 全NaN!" if r >= 1.0 else ""
        print(f"      {f:<14s} {r * 100:6.2f}{flag}")
    if (nan_ratio >= 1.0).any():
        bad = nan_ratio[nan_ratio >= 1.0].index.tolist()
        print(f"  [警告] 以下因子全为 NaN, 请检查: {bad}")


# ---------------------------------------------------------------- 主流程
def build_factor_panel(
    data_dir: str,
    dates: list = None,
    codes: list = None,
    workers: int = 1,
    industry_map: pd.Series = None,
    size: pd.DataFrame = None,
) -> pd.DataFrame:
    """构建因子面板并返回 MultiIndex [DataDate, SecuCode] 的 DataFrame。

    流程: build_panel -> add_apm -> preprocess(去极值+标准化), 不做平滑。
    dates 为 None 时自动扫描 data_dir 下全部交易日。
    """
    if dates is None:
        dates = scan_dates(data_dir)

    print(f"[1] build_panel: {len(dates)} 个交易日 ({dates[0]} ~ {dates[-1]})...")
    panel = build_panel(data_dir, dates, codes, workers=workers)
    print(f"    {panel['trading_day'].nunique()} 个交易日, "
          f"{panel['security_code'].nunique()} 只证券, {len(panel)} 行")

    print("[2] add_apm (滚动20日)...")
    panel = add_apm(panel)

    print("[3] preprocess (去极值 + 标准化)...")
    panel = preprocess(panel, industry_map=industry_map, size=size)

    # ---- 整理为 [DataDate, SecuCode] x features
    out = panel[["trading_day", "security_code"] + ALL_FACTORS].copy()
    out = out.rename(columns={"trading_day": "DataDate", "security_code": "SecuCode"})
    out = out.set_index(["DataDate", "SecuCode"]).sort_index()

    # 仅保留特征列, 强制 float
    out = out[ALL_FACTORS].astype("float64")

    _validate(out, ALL_FACTORS)
    return out


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="生成 [DataDate, SecuCode] x feature 因子面板 (parquet)")
    ap.add_argument("--data-dir", default="/data/cephfs/minute/one_minute_stat",
                    help="一分钟统计数据目录, 自动扫描全部 {yyyymmdd}.parquet")
    ap.add_argument("--out", default="factors_panel.parquet", help="输出 parquet 路径")
    ap.add_argument("--industry-file", default=None, help="行业映射 csv (index=code, 1列=行业)")
    ap.add_argument("--size-file", default=None, help="市值表 (trading_day, security_code, mktcap)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    # 输出格式固定为 parquet
    if not args.out.endswith(".parquet"):
        raise ValueError(f"输出仅支持 .parquet 格式, 收到: {args.out}")

    ind_map = None
    if args.industry_file:
        ind_map = pd.read_csv(args.industry_file, index_col=0).iloc[:, 0].astype(str)

    size_df = None
    if args.size_file:
        size_df = (pd.read_parquet(args.size_file) if args.size_file.endswith(".parquet")
                   else pd.read_csv(args.size_file, parse_dates=["trading_day"]))

    df = build_factor_panel(
        data_dir=args.data_dir,
        dates=None,                      # 自动扫描目录
        codes=None,                      # 保留全部证券
        workers=args.workers,
        industry_map=ind_map,
        size=size_df,
    )

    # 保存 (固定 parquet, DataDate 固定为 datetime)
    df.to_parquet(args.out)

    print(f"\n已保存: {args.out}  shape={df.shape}")
    print("预览:\n", df.head().round(4).to_string())


if __name__ == "__main__":
    main()