import pandas as pd 

# df = pd.read_parquet("_result/FreTSMixerIC_kline_[1d_half_hour_five_minute_one_minute]_spec_5_MLaT_lncap/factor_data/pred_EWM15.parquet")
df = pd.read_parquet("_result/GRUIC_kline_[1d_half_hour_five_minute_one_minute]_spec_5_MLaT_lncap/factor_data/pred_EWM15.parquet")

df.sort_values(['DataDate', 'SecuCode'], inplace=True)

df['pct_rank'] = df.groupby('DataDate')['pred_EWM15'].rank(pct=True)
df['weight'] = (df['pct_rank'] > 0.9).astype(float)
df['weight'] = df['weight'] / df.groupby('DataDate')['weight'].transform("sum")
df = df[['DataDate', 'SecuCode', 'weight']].reset_index(drop=True)
df['SecuCode'] = df['SecuCode'].astype(str).str.zfill(6)

df.to_parquet("GRU.parquet")