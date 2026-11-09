import argparse
import warnings

# 精准匹配：只有当警告类型是 UserWarning 且内容包含 "A worker stopped" 时才忽略
warnings.filterwarnings(
    "ignore", 
    category=UserWarning, 
    message=".*A worker stopped while some jobs were given to the executor.*"
)

from time import time
from data_loader.data_set_builder import build_kline_data_set # pylint: disable=W0611
from data_loader.data_set_builder import build_sig_data_set # pylint: disable=W0611
from data_loader.data_set_builder import build_Daily_hff_data_set # pylint: disable=W0611
# from data_loader._data_kline import build_ft_kline_min
# from data_loader._data_sig import build_ts_sig

def parse_args():
    parser = argparse.ArgumentParser(description="数据集构建工具")
    parser.add_argument(
        "--freq",
        type=str,
        default="1d",
        help="数据频率，例如: '1d', '5m', '1d_V3_4_ordo3' 等"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=80,
        help="回溯窗口长度，默认为 80"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start = time()

    # build_Daily_hff_data_set(
    #     lookback=args.lookback,
    #     data_path="data_raw/hff_v1/factors_panel.parquet"
    # )
    
    # build_sig_data_set(
    #     lookback=args.lookback,
    #     freq=args.freq
    # )

    build_kline_data_set(
        lookback=args.lookback,
        freq=args.freq
    )
    print(f"构建完成: freq={args.freq}, lookback={args.lookback}")

    # build_ts_sig("20150504", args.lookback, args.freq)

    print(f"耗时: {time() - start:.2f}s")
    