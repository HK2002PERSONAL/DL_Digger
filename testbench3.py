import pandas as pd 

# data = pd.read_parquet("data/test_result/GRU_1d_80_stock_5.parquet").set_index(["SecuCode", "DataDate"]).sort_index()

# data = data.groupby("SecuCode").ewm(15).mean().droplevel(0)

# data = data.swaplevel().sort_index()

# data.reset_index(drop=False).to_parquet("GRU_1d_80_stock_5_EWM15.parquet")

# freqs = ['one_minute', 'five_minute', 'half_hour', 'one_hour', '1d']

def zscore_add():
    sig = [
        pd.read_parquet("data/test_result/TCNIC_kline_multi_freq_spec_5_MLaT.parquet").set_index(['DataDate', "SecuCode"]).sort_index(),
        pd.read_parquet("data/test_result/GRUIC_kline_multi_freq_spec_5_MLaT.parquet").set_index(['DataDate', "SecuCode"]).sort_index(),
        # pd.read_parquet("data/test_result/GRUIC_five_minute_10_spec_5.parquet").set_index(['DataDate', "SecuCode"]).sort_index(),
        # pd.read_parquet("data/test_result/GRUIC_kline_one_minute_2_spec_5_MLaT.parquet").set_index(['DataDate', "SecuCode"]).sort_index(),
    ]

    sig = pd.concat(sig, axis=1, join="inner")
    mean = sig.groupby("DataDate").transform("mean")
    std = sig.groupby("DataDate").transform("std")

    sig = (sig - mean) / (std + 1e-8)

    sig = sig.mean(axis=1).to_frame("pred")

    sig.reset_index().to_parquet("GRUIC_MLaT_TCNIC_MLaT_ZscoreAdd.parquet")

def rank_add():
    sig = [
        pd.read_parquet("data/test_result/TCNIC_kline_multi_freq_spec_5_MLaT.parquet").set_index(['DataDate', "SecuCode"]).sort_index(),
        pd.read_parquet("data/test_result/GRUIC_kline_multi_freq_spec_5_MLaT.parquet").set_index(['DataDate', "SecuCode"]).sort_index(),
        # pd.read_parquet("data/test_result/GRUIC_five_minute_10_spec_5.parquet").set_index(['DataDate', "SecuCode"]).sort_index(),
        # pd.read_parquet("data/test_result/GRUIC_kline_one_minute_2_spec_5_MLaT.parquet").set_index(['DataDate', "SecuCode"]).sort_index(),
    ]

    sig = pd.concat(sig, axis=1, join="inner").groupby("DataDate").rank(pct=True)

    sig = sig.mean(axis=1).to_frame("pred")

    sig.reset_index().to_parquet("GRUIC_MLaT_TCNIC_MLaT_RankAdd.parquet")

if __name__ == "__main__":
    rank_add()

        