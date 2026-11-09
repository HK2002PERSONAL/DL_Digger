import pandas as pd

from utils.BackTester.back_testV2 import batch_test_multi_field

if __name__ == "__main__":

    Signal_Mark_1 = "GRU"
    Signal_Mark_2 = "FRETSMIXER"
    TEST_MARK = "TEST"

    SMOOTH_DECAY = 15

    signal_data_1 = pd.read_parquet("data/test_result/GRUIC_kline_[1d_half_hour_five_minute_one_minute]_spec_5_MLaT.parquet")
    signal_data_1 = signal_data_1.set_index(['SecuCode', 'DataDate']).sort_index()

    signal_data_2 = pd.read_parquet("data/test_result/FreTSMixerIC_kline_[1d_half_hour_five_minute_one_minute]_spec_5_MLaT.parquet")
    # signal_data_2['SecuCode'] = signal_data_2['SecuCode'].astype(int)
    signal_data_2 = signal_data_2.set_index(['SecuCode', 'DataDate']).sort_index()

    signal_data = pd.concat([
        signal_data_1['pred'], 
        signal_data_2['pred'],
        ], axis=1, join="inner")
    signal_data.columns = [
        Signal_Mark_1, 
        Signal_Mark_2,
    ]

    signal_data = signal_data.groupby("SecuCode").ewm(span=SMOOTH_DECAY).mean().droplevel(0)

    signal_data = signal_data[signal_data.index.get_level_values('DataDate') > pd.to_datetime("2021-12-31")]

    batch_test_multi_field(signal_data, f"{TEST_MARK}_cap", neutral_factor=['lncap', 'nls'])

    batch_test_multi_field(signal_data, f"{TEST_MARK}_allneut", 
        neutral_factor=['beta', 'bp', 'ey', 'growth', 'leverage', 'liquidity', 'lncap', 'momentum', 'nls', 'rv', 'contrarian', 'short_rev'])