import pandas as pd

from utils.BackTester.back_testV2 import batch_test

if __name__ == "__main__":

    SIGNAL_MARK = "FreTSMixerIC_kline_[1d_half_hour_five_minute_one_minute]_spec_5_MLaT"
    SIGNAL_PATH = "data/test_result/FreTSMixerIC_kline_[1d_half_hour_five_minute_one_minute]_spec_5_MLaT.parquet"

    signal_data = pd.read_parquet(SIGNAL_PATH)
    signal_data = signal_data.set_index(['SecuCode', 'DataDate']).sort_index()
    # signal_data = signal_data.set_index(['DataDate', 'SecuCode']).sort_index()
    # signal_data = signal_data.groupby("DataDate").rank(pct=True)
    # signal_data = signal_data.swaplevel().sort_index()

    # for smooth_decay in [5, 10, 15, 20, 60, 120, 240]:
    for smooth_decay in [5, 10, 15, 20]:
        signal_data[f'pred_EWM{smooth_decay}'] = signal_data.groupby("SecuCode")['pred'].ewm(span=smooth_decay).mean().droplevel(0)

    # signal_data = signal_data.groupby("SecuCode").ewm(span=15).mean().droplevel(0)


    signal_data = signal_data[signal_data.index.get_level_values('DataDate') > pd.to_datetime("2019-12-31")]

    batch_test(signal_data, f"{SIGNAL_MARK}_lncap", neutral_factor=['lncap'])

    batch_test(signal_data, f"{SIGNAL_MARK}_allneut", 
        neutral_factor=['beta', 'bp', 'ey', 'growth', 'leverage', 'liquidity', 'lncap', 'momentum', 'nls', 'rv', 'contrarian', 'short_rev'])