from utils.Logger import smart_log

import gc
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Literal, List, Callable, Optional

import warnings
warnings.filterwarnings("ignore", message=".*not writable.*", category=UserWarning)

import torch

from ._data_kline import load_ft_kline
from ._data_sig import load_ft_sig
from ._data_hff import load_ft_hff
from .__utils__ import TARGET_PATH, _ensure_dir

# ==============================
# 辅助函数
# ==============================

def _paralled_cal_by_feature(
    feature: np.ndarray,
    func: Callable,
    max_worker: int
) -> np.ndarray:
    """
    对大型特征矩阵沿特征维度并行计算统计量（mean/std 等）。
    用于 feature.shape[-1] > 32 时节省内存。

    返回形状 (1, 1, n_features)，与 feature.mean(axis=(0,1), keepdims=True) 一致。
    """
    n_features = feature.shape[-1]
    results = Parallel(n_jobs=max_worker, prefer='threads')(
        delayed(func)(feature[:, :, i])
        for i in range(n_features)
    )
    return np.array(results).reshape((1, 1, n_features))


def _valid_secu_mask(idx: pd.DataFrame) -> np.ndarray:
    """
    返回布尔 mask，标记 idx（以 ['SecuCode', 'DataDate'] 为索引）中
    SecuCode 是否为合法的纯数字字符串。

    源数据（尤其 sig 类特征索引）中可能混入空字符串 '' 或含 '\\x00'
    的损坏值，下游 SecuCode.astype(int) 会因此报错，需在此统一排除。
    """
    secu = pd.Series(idx.index.get_level_values('SecuCode')).astype(str)
    mask = secu.str.fullmatch(r'[0-9]+').fillna(False)
    return mask.to_numpy(dtype=bool)


def get_dataset(
    day_begin: str | int,
    day_end: str | int,
    lookback: int,
    ft_type: str, 
    freq: str,
    day_list: List[int],
    idx: pd.DataFrame,
    device: Optional[torch.device] = None,
    ft_mean: Optional[np.ndarray] = None,
    lb_mean: Optional[np.ndarray] = None,
    ft_std: Optional[np.ndarray] = None,
    lb_std: Optional[np.ndarray] = None,
    max_worker: int = 4,
    label_type: str = 'stock',
    label_span: int = 5,
    z_score_type: Literal['set', 'batch'] = 'batch',
    z_score_method: Literal['simple', 'robust'] = 'simple',
    test_mode: bool = False,
):
    """
    构建一个时间区间的数据集（X/y/索引），并可执行 set 级别标准化。

    参数
    ----
    - day_begin, day_end : str 日期范围，格式 'YYYYMMDD'
    - lookback : int 回看交易日数量
    - freq : str 数据频率
    - day_list : List[str] 可用交易日列表
    - idx : pd.DataFrame 索引表，含 ['DataDate', 'SecuCode'] 列
    - device : Optional[torch.device] GPU 设备（用于 z-score 加速）
    - ft_mean, ft_std, lb_mean, lb_std : Optional[np.ndarray] 标准化参数
    - max_worker : int 并行进程数
    - label_type : str 标签类型
    - z_score_type : 'set' 或 'batch'
    - z_score_method : 'simple' 或 'robust'
    - test_mode : bool 测试模式不处理 y

    返回
    ----
    feature, label, idx, ft_mean, ft_std, lb_mean, lb_std
    """
    smart_log('loading data')

    day_begin, day_end = int(day_begin), int(day_end)

    daylist = [i for i in day_list if i <= day_end and i >= day_begin]
    smart_log(f'given date length: {len(daylist)}')

    idx['DataDate'] = idx['DataDate'].astype(int)
    idx = idx.loc[(idx['DataDate'] <= day_end) & (idx['DataDate'] >= day_begin)].copy(deep=True)
    smart_log(f'chosen date length: {len(idx["DataDate"].unique())}')

    idx['origin_idx'] = np.arange(len(idx), dtype=np.int32)
    idx['DataDate'] = pd.to_datetime(idx['DataDate'], format="%Y%m%d")
    idx.set_index(['SecuCode', 'DataDate'], inplace=True)

    # 并行加载每个交易日特征
    if ft_type == "kline":
        load_ft = load_ft_kline
    elif ft_type == "sig":
        load_ft = load_ft_sig
    elif ft_type == "hff":
        load_ft = load_ft_hff
    else:
        raise NotImplementedError("暂不支持此数据格式")
    
    data_list = Parallel(n_jobs=max_worker)(
        delayed(load_ft)(day, lookback, freq)
        for day in tqdm(daylist, desc='Loading feature', leave=False)
    )

    lengths = [arr.shape[0] for arr in data_list]
    total_shape = (sum(lengths),) + data_list[0].shape[1:]

    feature = np.zeros(total_shape, dtype=data_list[0].dtype)
    offsets = [0] + np.cumsum(lengths).tolist()[:-1]

    def _worker(i: int):
        np.copyto(feature[offsets[i]: offsets[i] + lengths[i]], data_list[i])

    Parallel(n_jobs=max_worker, prefer='threads')(
        delayed(_worker)(i)
        for i in tqdm(range(len(data_list)), desc='Parallel Filling', leave=False)
    )

    del data_list
    smart_log(f'feature shape: {feature.shape}, idx shape: {idx.shape}')

    if not test_mode:
        label = pd.read_feather(os.path.join(TARGET_PATH, 'label', f'{label_type}_ret.ftr')).set_index(['SecuCode', 'DataDate']).sort_index()

        smart_log('matching data')
        label = pd.concat([idx, label], axis=1, join='inner').dropna()
        smart_log(f'label matched: {label.shape}')

        idx = label[['origin_idx']]
        label = label[f'Return{label_span}d'].values.copy()

    else:
        idx = idx[['origin_idx']]

    # set 级标准化
    if z_score_type == 'set':
        smart_log('adjusting data')

        if z_score_method == 'robust':
            # 特征 robust z-score
            if ft_mean is None or ft_std is None:
                q10, ft_mean, q90 = np.quantile(feature, [0.1, 0.5, 0.9], axis=(0, 1), keepdims=True)
                smart_log(f'x mean: {ft_mean}')
                ft_std = q90 - q10
                smart_log(f'x std: {ft_std}')

            feature -= ft_mean
            feature /= (ft_std + 1e-12)
            np.clip(feature, -5, 5, out=feature)
            smart_log('x done')

        elif z_score_method == 'simple':
            # 特征 simple z-score
            # 注意：新数据（6列）已在 build_ft_kline_min 中做过 log1p，
            # 此处不再对 volume/value 列做 log1p
            if device is not None:
                feature = torch.from_numpy(feature).to(torch.float32).to(device)

                if ft_mean is None:
                    ft_mean = feature.mean(dim=(0, 1), keepdim=True)
                else:
                    ft_mean = torch.from_numpy(ft_mean).to(torch.float32).to(device)
                smart_log(f'x mean: {ft_mean}')

                if ft_std is None:
                    ft_std = feature.std(dim=(0, 1), keepdim=True)
                else:
                    ft_std = torch.from_numpy(ft_std).to(torch.float32).to(device)
                smart_log(f'x std: {ft_std}')

                feature.subtract_(ft_mean)
                feature.divide_(ft_std + 1e-12)
                feature.clip_(-5, 5)

                feature = feature.cpu().detach().numpy()
                ft_mean = ft_mean.cpu().detach().numpy()
                ft_std = ft_std.cpu().detach().numpy()
                smart_log('x done')
            else:
                if ft_mean is None:
                    if feature.shape[-1] > 32:
                        ft_mean = _paralled_cal_by_feature(feature, np.mean, max_worker)
                    else:
                        ft_mean = feature.mean(axis=(0, 1), keepdims=True)
                smart_log(f'x mean: {ft_mean}')

                if ft_std is None:
                    if feature.shape[-1] > 32:
                        ft_std = _paralled_cal_by_feature(feature, np.std, max_worker)
                    else:
                        ft_std = feature.std(axis=(0, 1), keepdims=True)
                smart_log(f'x std: {ft_std}')

                feature -= ft_mean
                feature /= (ft_std + 1e-12)
                np.clip(feature, -5, 5, feature)
                smart_log('x done')

        # 训练/验证时处理 y；测试模式不处理 y
        if not test_mode:
            lb_mean = label.mean() if lb_mean is None else lb_mean
            smart_log(f'y mean: {lb_mean}')

            lb_std = label.std() if lb_std is None else lb_std
            smart_log(f'y std: {lb_std}')

            label -= lb_mean
            label /= (lb_std + 1e-12)
            np.clip(label, -5, 5, label)
            smart_log('y done')

    if test_mode:
        label = None

    return feature, label, idx, ft_mean, ft_std, lb_mean, lb_std


def get_1x1y_dataset(
    train_day_begin: str,
    train_day_end: str,
    valid_day_begin: str,
    valid_day_end: str,
    test_day_begin: str,
    test_day_end: str,
    lookback: int,
    ft_type: Literal['kline', 'sig', 'hff'],
    freq: Literal['one_minute', 'five_minute', 'half_hour', 'one_hour', '1d'],
    device: Optional[torch.device] = None,
    max_worker: int = 32,
    label_type: str = 'stock',
    label_span: int = 5,
    z_score_type: Literal['set', 'batch'] = 'batch',
):
    """
    获取单输入单标签训练/验证/测试数据集，带缓存逻辑。
    - 训练/验证：返回 X/y/idx
    - 测试：仅返回 X/idx（不需要 y）

    返回
    ----
    train_x_data, train_y_data, train_idx,
    valid_x_data, valid_y_data, valid_idx,
    test_x_data, test_idx
    """
    assert (train_day_begin < train_day_end < valid_day_begin < valid_day_end < test_day_begin < test_day_end), \
        'date setting must be in order!'

    CACHE_PATH = os.path.join(
        TARGET_PATH,
        'cache_data',
        f'{ft_type}_{freq}_{lookback}_{z_score_type}-zScore_'
        f'{train_day_begin}-{train_day_end}_{valid_day_begin}-{valid_day_end}_{test_day_begin}-{test_day_end}'
    )

    X_TRAIN_CACHE_PATH = os.path.join(CACHE_PATH, 'x_train.npy')
    Y_TRAIN_CACHE_PATH = os.path.join(CACHE_PATH, f'{label_type}_{label_span}_train.npy')
    X_VALID_CACHE_PATH = os.path.join(CACHE_PATH, 'x_valid.npy')
    Y_VALID_CACHE_PATH = os.path.join(CACHE_PATH, f'{label_type}_{label_span}_valid.npy')

    # test only X
    X_TEST_CACHE_PATH = os.path.join(CACHE_PATH, 'x_test.npy')

    X_MEAN_CACHE_PATH = os.path.join(CACHE_PATH, 'x_mean.npy')
    X_STD_CACHE_PATH  = os.path.join(CACHE_PATH, 'x_std.npy')

    TRAIN_INDEX_PATH = os.path.join(CACHE_PATH, 'train.idx')
    VALID_INDEX_PATH = os.path.join(CACHE_PATH, 'valid.idx')
    TEST_INDEX_PATH  = os.path.join(CACHE_PATH, 'test.idx')

    # 1) 全缓存命中（train/valid/test）
    if (
        os.path.exists(X_TRAIN_CACHE_PATH) and
        os.path.exists(Y_TRAIN_CACHE_PATH) and
        os.path.exists(X_VALID_CACHE_PATH) and
        os.path.exists(Y_VALID_CACHE_PATH) and
        os.path.exists(X_TEST_CACHE_PATH) and
        os.path.exists(TRAIN_INDEX_PATH) and
        os.path.exists(VALID_INDEX_PATH) and
        os.path.exists(TEST_INDEX_PATH)
    ):
        smart_log('cache found')

        train_x_data = np.load(X_TRAIN_CACHE_PATH, mmap_mode='r')
        train_y_data = np.load(Y_TRAIN_CACHE_PATH, mmap_mode='r')
        valid_x_data = np.load(X_VALID_CACHE_PATH, mmap_mode='r')
        valid_y_data = np.load(Y_VALID_CACHE_PATH, mmap_mode='r')
        test_x_data  = np.load(X_TEST_CACHE_PATH, mmap_mode='r')

        # feather 不支持 MultiIndex，读取后需重建
        train_idx = pd.read_feather(TRAIN_INDEX_PATH)
        train_idx.set_index(['SecuCode', 'DataDate'], inplace=True)
        valid_idx = pd.read_feather(VALID_INDEX_PATH)
        valid_idx.set_index(['SecuCode', 'DataDate'], inplace=True)
        test_idx = pd.read_feather(TEST_INDEX_PATH)
        test_idx.set_index(['SecuCode', 'DataDate'], inplace=True)

    # 2) 仅 X 缓存命中（train/valid/test），重构 y
    elif (
        os.path.exists(X_TRAIN_CACHE_PATH) and
        os.path.exists(X_VALID_CACHE_PATH) and
        os.path.exists(X_TEST_CACHE_PATH) and
        os.path.exists(TRAIN_INDEX_PATH) and
        os.path.exists(VALID_INDEX_PATH) and
        os.path.exists(TEST_INDEX_PATH)
    ):
        smart_log('x cache found')

        if label_type.endswith('abs'):
            label_type = label_type[:-4]
            abs_flag = True
        else:
            abs_flag = False

        train_x_data = np.load(X_TRAIN_CACHE_PATH, mmap_mode='r')
        valid_x_data = np.load(X_VALID_CACHE_PATH, mmap_mode='r')
        test_x_data  = np.load(X_TEST_CACHE_PATH, mmap_mode='r')

        train_idx = pd.read_feather(TRAIN_INDEX_PATH)
        train_idx.set_index(['SecuCode', 'DataDate'], inplace=True)
        valid_idx = pd.read_feather(VALID_INDEX_PATH)
        valid_idx.set_index(['SecuCode', 'DataDate'], inplace=True)
        test_idx = pd.read_feather(TEST_INDEX_PATH)
        test_idx.set_index(['SecuCode', 'DataDate'], inplace=True)

        y_data = pd.read_feather(os.path.join(TARGET_PATH, 'label', f'{label_type}_ret.ftr')) \
            .set_index(['SecuCode', 'DataDate'])

        train_y_df = pd.concat([train_idx, y_data], axis=1, join='inner').dropna()
        smart_log(f'label matched(train): {train_y_df.shape}')

        valid_y_df = pd.concat([valid_idx, y_data], axis=1, join='inner').dropna()
        smart_log(f'label matched(valid): {valid_y_df.shape}')

        train_idx = train_y_df[['origin_idx']]
        train_y_data = train_y_df[f'Return{label_span}d'].values.copy()

        valid_idx = valid_y_df[['origin_idx']]
        valid_y_data = valid_y_df[f'Return{label_span}d'].values.copy()

        lb_mean = train_y_data.mean()
        smart_log(f'y mean: {lb_mean}')
        lb_std = train_y_data.std()
        smart_log(f'y std: {lb_std}')

        if abs_flag:
            np.clip(train_y_data, lb_mean - 5 * lb_std, lb_mean + 5 * lb_std, train_y_data)
            np.clip(valid_y_data, lb_mean - 5 * lb_std, lb_mean + 5 * lb_std, valid_y_data)
        else:
            train_y_data -= lb_mean
            train_y_data /= (lb_std + 1e-12)
            np.clip(train_y_data, -5, 5, train_y_data)

            valid_y_data -= lb_mean
            valid_y_data /= (lb_std + 1e-12)
            np.clip(valid_y_data, -5, 5, valid_y_data)

    # 3) 无缓存，完整构建（包含 test 的 X-only）
    else:
        smart_log('cache not found')

        idx = pd.read_feather(os.path.join(TARGET_PATH, f'{ft_type}_{freq}_lag{lookback}.feather'))
        daylist = sorted(idx['DataDate'].unique().tolist())
        daylist = [int(i) for i in daylist]

        _ensure_dir(CACHE_PATH)

        smart_log('building training data')
        train_x_data, train_y_data, train_idx, x_mean, x_std, y_mean, y_std = get_dataset(
            train_day_begin, train_day_end,
            lookback,
            ft_type,
            freq,
            daylist, idx, device,
            max_worker=max_worker,
            label_type=label_type, label_span=label_span,
            z_score_type=z_score_type
        )

        if z_score_type == 'set':
            np.save(X_MEAN_CACHE_PATH, x_mean)
            smart_log('x mean done')
            np.save(X_STD_CACHE_PATH, x_std)
            smart_log('x std done')

        np.save(X_TRAIN_CACHE_PATH, train_x_data)
        smart_log('x train done')
        np.save(Y_TRAIN_CACHE_PATH, train_y_data)
        smart_log('y train done')
        train_idx.reset_index().to_feather(TRAIN_INDEX_PATH)
        smart_log('train idx done')

        # 释放 train 大数组，以 mmap 重新读取
        del train_x_data, train_y_data
        gc.collect()
        train_x_data = np.load(X_TRAIN_CACHE_PATH, mmap_mode='r')
        train_y_data = np.load(Y_TRAIN_CACHE_PATH, mmap_mode='r')

        smart_log('building valid data')
        valid_x_data, valid_y_data, valid_idx, _, _, _, _ = get_dataset(
            valid_day_begin, valid_day_end,
            lookback,
            ft_type,
            freq,
            daylist, idx, device,
            ft_mean=x_mean, lb_mean=y_mean, ft_std=x_std, lb_std=y_std,
            max_worker=max_worker,
            label_type=label_type, label_span=label_span,
            z_score_type=z_score_type
        )

        np.save(X_VALID_CACHE_PATH, valid_x_data)
        smart_log('x valid done')
        np.save(Y_VALID_CACHE_PATH, valid_y_data)
        smart_log('y valid done')
        valid_idx.reset_index().to_feather(VALID_INDEX_PATH)
        smart_log('valid idx done')

        # 释放 valid 大数组，以 mmap 重新读取
        del valid_x_data, valid_y_data
        gc.collect()
        valid_x_data = np.load(X_VALID_CACHE_PATH, mmap_mode='r')
        valid_y_data = np.load(Y_VALID_CACHE_PATH, mmap_mode='r')

        smart_log('building test data (X only)')
        test_x_data, _, test_idx, _, _, _, _ = get_dataset(
            test_day_begin, test_day_end,
            lookback,
            ft_type,
            freq,
            daylist, idx, device,
            ft_mean=x_mean, ft_std=x_std,  # 当 z_score_type=='set' 时用训练集统计量
            max_worker=max_worker,
            label_type=label_type, label_span=label_span,
            z_score_type=z_score_type,
            test_mode=True
        )

        np.save(X_TEST_CACHE_PATH, test_x_data)
        smart_log('x test done')
        test_idx.reset_index().to_feather(TEST_INDEX_PATH)
        smart_log('test idx done')

        # 释放 test 大数组，以 mmap 重新读取
        del test_x_data
        gc.collect()
        test_x_data = np.load(X_TEST_CACHE_PATH, mmap_mode='r')

    # 排除非法 SecuCode（空字符串 / 含 \x00 的损坏值），
    # 防止下游 SecuCode.astype(int) 报错；
    # y 与 idx 逐行对齐，需同步过滤；x 通过 origin_idx 间接引用，无需过滤。
    train_mask = _valid_secu_mask(train_idx)
    if not train_mask.all():
        smart_log(f'filtering invalid SecuCode in train: {(~train_mask).sum()}')
        train_idx = train_idx[train_mask]
        train_y_data = np.asarray(train_y_data)[train_mask]

    valid_mask = _valid_secu_mask(valid_idx)
    if not valid_mask.all():
        smart_log(f'filtering invalid SecuCode in valid: {(~valid_mask).sum()}')
        valid_idx = valid_idx[valid_mask]
        valid_y_data = np.asarray(valid_y_data)[valid_mask]

    test_mask = _valid_secu_mask(test_idx)
    if not test_mask.all():
        smart_log(f'filtering invalid SecuCode in test: {(~test_mask).sum()}')
        test_idx = test_idx[test_mask]

    return (
        train_x_data, train_y_data, train_idx,
        valid_x_data, valid_y_data, valid_idx,
        test_x_data, test_idx
    )



def get_1xhy_dataset(
    train_day_begin: str,
    train_day_end: str,
    valid_day_begin: str,
    valid_day_end: str,
    test_day_begin: str,
    test_day_end: str,
    lookback: int,
    ft_type: Literal['kline', 'sig'],
    freq: Literal['one_minute', 'five_minute', 'half_hour', 'one_hour', '1d'],
    horizon: int, 
    label_type: str = 'stock',
    label_span: int = 1,
    z_score_type: Literal['set', 'batch'] = 'set',
):
    """
    获取日频K线数据集（单输入 + horizon 标签）。

    特征(X) 复用 get_1x1y_dataset 的缓存，若缓存不存在则直接报错。
    标签(Y) 仅做 clip（mean ± 5*std），不做 normalize；
    y_mean / y_std 一并返回，供用户在损失函数中自行决定是否 normalize。

    Parameters
    ----------
    train_day_begin ~ test_day_end : str
        日期范围
    lookback : int
        特征回溯窗口大小
    horizon : int
        标签 horizon（未来天数）
    z_score_type : str
        标准化类型，需与 get_1x1y_dataset 一致

    Returns
    -------
    train_x_data, train_y_data, train_idx,
    valid_x_data, valid_y_data, valid_idx,
    test_x_data, test_idx,
    """
    assert (train_day_begin < train_day_end < valid_day_begin < valid_day_end < test_day_begin < test_day_end), \
        'date setting must be in order!'

    # ── X 缓存路径：与 get_1x1y_dataset 完全一致 ──
    X_CACHE_PATH = os.path.join(
        TARGET_PATH, 'cache_data',
        f'{ft_type}_{freq}_{lookback}_{z_score_type}-zScore_'
        f'{train_day_begin}-{train_day_end}_{valid_day_begin}-{valid_day_end}_{test_day_begin}-{test_day_end}'
    )

    # X cache files (from get_1x1y_dataset)
    X_TRAIN_CACHE = os.path.join(X_CACHE_PATH, 'x_train.npy')
    X_VALID_CACHE = os.path.join(X_CACHE_PATH, 'x_valid.npy')
    X_TEST_CACHE  = os.path.join(X_CACHE_PATH, 'x_test.npy')
    X_TRAIN_IDX   = os.path.join(X_CACHE_PATH, 'train.idx')
    X_VALID_IDX   = os.path.join(X_CACHE_PATH, 'valid.idx')
    X_TEST_IDX    = os.path.join(X_CACHE_PATH, 'test.idx')

    # ============================================================
    # Step 1: 检查并加载 X 缓存（必须存在）
    # ============================================================
    required_x_files = [X_TRAIN_CACHE, X_VALID_CACHE, X_TEST_CACHE,
                        X_TRAIN_IDX, X_VALID_IDX, X_TEST_IDX]
    if not all(os.path.exists(f) for f in required_x_files):
        raise FileNotFoundError(
            f"X cache not found at {X_CACHE_PATH}. "
            f"Please run get_1x1y_dataset first with ft_type='{ft_type}', "
            f"freq='{freq}', lookback={lookback}, z_score_type='{z_score_type}', "
            f"and the same date range."
        )

    smart_log(f'loading X cache from {X_CACHE_PATH}')
    train_x_data = np.load(X_TRAIN_CACHE, mmap_mode='r')
    valid_x_data = np.load(X_VALID_CACHE, mmap_mode='r')
    test_x_data = np.load(X_TEST_CACHE, mmap_mode='r')

    train_idx = pd.read_feather(X_TRAIN_IDX)
    train_idx.set_index(['SecuCode', 'DataDate'], inplace=True)
    valid_idx = pd.read_feather(X_VALID_IDX)
    valid_idx.set_index(['SecuCode', 'DataDate'], inplace=True)
    test_idx = pd.read_feather(X_TEST_IDX)
    test_idx.set_index(['SecuCode', 'DataDate'], inplace=True)
    smart_log("X data loaded")

    # ============================================================
    # Step 2: 制作horizon Y
    # ============================================================
    y_data = pd.read_feather(os.path.join(TARGET_PATH, 'label', f'{label_type}_ret.ftr'))
    y_data = y_data.set_index(['SecuCode', 'DataDate'])[f'Return{label_span}d'].sort_index()
    y_data = pd.concat({
        f"h{i}":y_data.groupby("SecuCode").shift(-i) for i in range(horizon)
    }, axis=1, join="inner").dropna().sort_index(level=[1,0])
    smart_log("Horizon data built")

    train_avail_idx = train_idx.index.intersection(y_data.index)
    valid_avail_idx = valid_idx.index.intersection(y_data.index)

    smart_log(f"Data Matched: train {len(train_avail_idx)}; valid {len(valid_avail_idx)}")

    train_y_data = y_data.loc[train_avail_idx].values
    valid_y_data = y_data.loc[valid_avail_idx].values

    train_idx = train_idx.loc[train_avail_idx]
    valid_idx = valid_idx.loc[valid_avail_idx]

    # 排除非法 SecuCode（train/valid 已与 y_data 求交集，天然干净；此处仅处理 test）
    test_mask = _valid_secu_mask(test_idx)
    if not test_mask.all():
        smart_log(f'filtering invalid SecuCode in test: {(~test_mask).sum()}')
        test_idx = test_idx[test_mask]

    return (
        train_x_data, train_y_data, train_idx,
        valid_x_data, valid_y_data, valid_idx,
        test_x_data, test_idx,
    )
