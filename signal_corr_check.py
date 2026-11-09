"""
检查两个因子预测值之间的日度截面相关性
- Pearson 截面相关 (daily IC 口径)
- Spearman 截面相关 (rankIC 口径)
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

FILE_A = "data/test_result/GRUIC_one_minute_2_spec_5.parquet"
FILE_B = "data/test_result/TCNIC_one_minute_2_spec_5.parquet"

def main():
    df_a = pd.read_parquet(FILE_A); print(df_a.info())
    df_b = pd.read_parquet(FILE_B); df_b['SecuCode'] = df_b['SecuCode'].astype(int)

    # 按 (DataDate, SecuCode) 合并，确保对齐
    merged = df_a.merge(df_b, on=["DataDate", "SecuCode"], suffixes=("_A", "_B"))
    merged.columns = ["DataDate", "SecuCode", "pred_A", "pred_B"]
    merged = merged.dropna(subset=["pred_A", "pred_B"])

    print(f"文件A: {FILE_A}, 行数: {len(df_a)}")
    print(f"文件B: {FILE_B}, 行数: {len(df_b)}")
    print(f"合并后行数: {len(merged)}, 日期数: {merged['DataDate'].nunique()}")
    print(f"日期范围: {merged['DataDate'].min()} ~ {merged['DataDate'].max()}")
    print()

    # 按日计算截面相关性
    results = []
    for date, grp in merged.groupby("DataDate"):
        if len(grp) < 10:
            continue
        x, y = grp["pred_A"].values, grp["pred_B"].values
        pearson_corr = np.corrcoef(x, y)[0, 1]
        spearman_corr, _ = spearmanr(x, y)
        results.append({
            "DataDate": date,
            "stock_count": len(grp),
            "pearson_corr": pearson_corr,
            "spearman_corr": spearman_corr,
        })

    corr_df = pd.DataFrame(results).set_index("DataDate")

    # 按年汇总
    corr_df["year"] = pd.to_datetime(corr_df.index.astype(str)).year
    yearly = corr_df.groupby("year")[["pearson_corr", "spearman_corr"]].mean()

    # 构建汇总表: 左列 pearson, 右列 spearman
    # 行: 全局平均 / 中位数 / 最大相关系数 / 分年平均
    summary = pd.DataFrame(
        {
            "pearson_corr": [
                corr_df["pearson_corr"].mean(),
                corr_df["pearson_corr"].median(),
                corr_df["pearson_corr"].max(),
            ],
            "spearman_corr": [
                corr_df["spearman_corr"].mean(),
                corr_df["spearman_corr"].median(),
                corr_df["spearman_corr"].max(),
            ],
        },
        index=["mean", "median", "max"],
    )

    yearly.index = yearly.index.astype(str)
    full_table = pd.concat([summary, yearly])
    full_table.index.name = "stat"

    print("=" * 60)
    print("日度截面相关性统计")
    print("=" * 60)
    print(f"有效交易日数: {len(corr_df)}")
    print(f"平均股票数: {corr_df['stock_count'].mean():.0f}")
    print()
    print(full_table.round(6).to_markdown())

    # 保存明细
    corr_df.to_csv("factor_corr_result.csv")
    print("\n明细已保存至 factor_corr_result.csv")

if __name__ == "__main__":
    main()