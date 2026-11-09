import os
import json
from typing import Tuple, List, Literal

import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from .utils.run_cmd import run_shell_cmd_with_file_addition_check
from .utils.pdf_processor import merge_all_pdfs

def format_testor(name: str, neutral_factor: List[str], field: str):

    # 创建新的config
    with open(os.path.join('utils', 'BackTester', 'config', 'config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 默认中性化lncap
    config["neutralize"] = ['lncap'] if neutral_factor is None else neutral_factor

    config["universe"] = field

    with open(os.path.join(name, f'{name}config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    # 创建新的batchconfig
    with open(os.path.join('utils', 'BackTester', 'config', 'batchconfig.json'), 'r', encoding='utf-8') as f:
        batchconfig = json.load(f)

    batchconfig['config_path'] = os.path.join(name, f'{name}config.json')
    batchconfig['data_directory'] = os.path.join(name, 'factor_data')
    batchconfig['foldername'] = os.path.join(name, f"factor_result_{field}")

    batchconfig_file = os.path.join(name, f'{name}batchconfig.json')
    with open(batchconfig_file, 'w', encoding='utf-8') as f:
        json.dump(batchconfig, f, indent=4, ensure_ascii=False)

    return f"OPENBLAS_NUM_THREADS=4 sigtest -b -f {batchconfig_file}"


def get_good_factor(test_result: pd.DataFrame, ic_threshold: float = 0.01):

    test_result.index = test_result['filename'].str.split('/').str[-2].str.split('.').str[0]

    trend_pass = test_result[['hedge_ir', 'long_ir', 'short_ir', 'ic_mean']].apply(np.sign).nunique(axis=1) == 1
    ic_pass = test_result['ic_mean'].abs() >= ic_threshold

    return test_result.loc[trend_pass & ic_pass].index.to_list()


def batch_test(
    data: pd.DataFrame, 
    name: str, 
    neg_corr: bool = False, 
    winsorize: bool = False,
    smooth_decay: int = 0, 
    neutral_factor: List[str] = None,
    ic_threshold: float = 0.01,
    field: Literal["allmarket", "hs300", "zz500", "zz1000"] = "allmarket"
):

    if neutral_factor is None:
        neutral_factor = ['lncap']

    if smooth_decay:
        data = data.groupby("SecuCode").ewm(span=smooth_decay).mean().droplevel(0)
        name = name + f"_EWM{smooth_decay}"

    if neg_corr:
        data = -data
        name = name + "_NEG"

    data = data.swaplevel().sort_index()

    if winsorize:
        data_group = data.groupby("DataDate")
        data_mean = data_group.transform("mean")
        data_std = data_group.transform("std")
        data = (data - data_mean) / (data_std + 1e-10)
        data.clip(-5, 5, inplace=True)
        name = name + "_WS"

    result_field = f"factor_result_{field}"

    os.makedirs(name, exist_ok=True)
    os.makedirs(os.path.join(name, 'factor_data'), exist_ok=True)
    os.makedirs(os.path.join(name, result_field), exist_ok=True)

    def _format_factor(factor_name):
        _factor = data[[factor_name]].dropna()
        _valid_date = _factor.groupby('DataDate')[factor_name].transform("count")
        _factor.loc[_valid_date > 9, [factor_name]].reset_index(drop=False).to_parquet(
            os.path.join(name, 'factor_data', f'{factor_name}.parquet'))

    Parallel(n_jobs=4)(
        delayed(_format_factor)(factor_name) for factor_name in tqdm(data.columns, desc="Preparing Data")
    )

    print("Running BackTest")
    expected_folders = os.listdir(os.path.join(name, "factor_data"))
    run_shell_cmd_with_file_addition_check(name, format_testor(name, neutral_factor, field), os.path.join(name, result_field), expected_folders)

    print("Arranging")
    merge_all_pdfs(os.path.join(name, result_field), os.path.join(name, result_field, f"{name}.pdf"))

    result = pd.read_csv(os.path.join(name, result_field, "state_factor_return.csv"))
    result.index = result['filename'].str.split('/').str[-2].str.split('.').str[0]
    good_factors = get_good_factor(result, ic_threshold)
    result.loc[good_factors].sort_index().to_csv(os.path.join(name, f"{name}.csv"))
    
    return data[good_factors]


def batch_test_multi_field(
    data: pd.DataFrame, 
    name: str, 
    neg_corr: bool = False, 
    winsorize: bool = False,
    smooth_decay: int = 0, 
    neutral_factor: List[str] = None,
    ic_threshold: float = 0.01,
    fields: Tuple[str] = ("allmarket", "hs300", "zz500", "zz1000")
):
    """multi field test"""
    for field in fields:
        print(f"\n===back testing {field}===\n")
        batch_test(data, name, neg_corr, winsorize, smooth_decay, neutral_factor, ic_threshold, field)