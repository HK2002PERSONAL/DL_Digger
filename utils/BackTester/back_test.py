import os
import time
import json
import string
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import product
from pypdf import PdfWriter, PdfReader


def format_factor(data: pd.DataFrame, factor_name: str, path: os.PathLike):
    data[[factor_name]].reset_index(drop=False).to_parquet(os.path.join(path, f'{factor_name}.parquet'))


def format_testor(name):
    with open('batchconfig.json') as f:
        config = json.load(f)

    config['data_directory'] = os.path.join(name, 'factor_data')
    config['foldername'] = os.path.join(name, 'factor_result')

    with open(f'{name}batchconfig.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    return f"OPENBLAS_NUM_THREADS=4 sigtest -b -f {name}batchconfig.json"


def run_shell_cmd(name, cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    tqdm.write(f"{name}: {cmd}\noutput: {result.stdout.strip()}") 


def run_shell_cmd_with_progress(name, cmd, target_path, expected_folders, appendix_num: int = 5):
    """
    name: 任务名称
    cmd: 执行命令
    target_path: 监控的路径
    expected_folders: 预期的文件夹名称列表（List）
    """
    total = len(expected_folders) + appendix_num
    expected_set = set(expected_folders)
    
    # 1. 非阻塞启动子进程
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # 2. 构建进度条
    with tqdm(total=total, desc=f"任务: {name}") as pbar:
        last_count = 0
        
        # 3. 轮询检查进度
        while process.poll() is None:  # process.poll() 为 None 表示进程仍在运行
            # 扫描目录下已存在的文件夹  
            current_folders = [d for d in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, d))]
            # 计算符合预期的文件夹交集数量
            current_count = len(expected_set.intersection(current_folders))
            
            # 更新进度条差值
            if current_count > last_count:
                pbar.update(current_count - last_count)
                last_count = current_count
            
            # 每 10 秒检查一次
            time.sleep(10)
            
        # 4. 进程结束后，确保进度条满格（处理最后一次扫描到结束之间的间隙）
        if last_count < total:
            pbar.update(total - last_count)

    # 获取最终输出
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        tqdm.write(f"✅ {name} 完成: {stdout}")
    else:
        tqdm.write(f"❌ {name} 失败: {stderr}")


def add_bookmarks_in_range(outline, writer, start, end, reader, parent=None):
    """
    递归处理目录：如果目录指向的页码在当前拆分范围内，则修正页码并添加到新文件中。
    """
    last_added_item = None
    for item in outline:
        # 如果是列表，说明是子目录（层级结构）
        if isinstance(item, list):
            if last_added_item: # 只有当父级目录在范围内时才添加子目录
                add_bookmarks_in_range(item, writer, start, end, reader, parent=last_added_item)
            continue
        
        # 获取该目录项指向的原始页码
        try:
            page_no = reader.get_destination_page_number(item)
            # 如果原页码在当前拆分的 [start, end) 范围内
            if start <= page_no < end:
                # 计算在新 PDF 中的相对偏移页码
                new_page_no = page_no - start
                # 将书签添加到新文件，并保存引用供递归使用
                last_added_item = writer.add_outline_item(item.title, new_page_no, parent=parent)
        except Exception:
            continue
        

def split_large_pdf(input_path, step=20):
    """
    如果文件超过 10MB，按 step 步长拆分，并保留对应范围的目录。
    """
    file_size = os.path.getsize(input_path)
    limit_size = 10 * 1024 * 1024  # 10MB
    
    if file_size <= limit_size:
        print(f"无需拆分: {input_path} ({file_size/1024/1024:.2f}MB)")
        return

    try:
        print(f"检测到大文件，开始拆分并迁移目录: {input_path}")
        reader = PdfReader(input_path)
        total_pages = len(reader.pages)
        file_prefix = os.path.splitext(input_path)[0]
        alphabet = string.ascii_lowercase
        
        # 获取原始文档的完整目录树
        full_outline = reader.outline

        for i, start_page in enumerate(range(0, total_pages, step)):
            writer = PdfWriter()
            end_page = min(start_page + step, total_pages)
            
            # 添加页面（使用 append 语法兼容性最好且保留内部资源）
            writer.append(reader, pages=(start_page, end_page))
            
            # 显式迁移并修正当前页码范围内的目录
            add_bookmarks_in_range(full_outline, writer, start_page, end_page, reader)

            # 命名逻辑
            suffix = alphabet[i] if i < 26 else str(i)
            output_path = f"{file_prefix}_PART{suffix}.pdf"
            
            with open(output_path, "wb") as f:
                writer.write(f)
            print(f"  -> 已生成: {os.path.basename(output_path)}")
            
    except Exception as e:
        print(f"拆分过程中出错: {e}")



def merge_all_pdfs(root_dir, output_filename="merged_output.pdf", split_pdf: bool = False):
    """
    遍历 root_dir 及其子目录，找到所有 .pdf 文件并合并。
    """
    merger = PdfWriter()
    pdf_files = []

    # 1. 递归遍历目录寻找 pdf
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                # 记录完整路径
                full_path = os.path.join(root, file)
                pdf_files.append(full_path)

    # 2. 对文件路径进行排序（确保合并顺序可控）
    pdf_files.sort()

    if not pdf_files:
        print("未在目录下找到任何 PDF 文件。")
        return

    # 3. 执行合并
    print(f"正在合并 {len(pdf_files)} 个文件...")
    try:
        for pdf in pdf_files:
            # 获取文件名作为书签名称（去掉扩展名）
            bookmark_name = os.path.basename(pdf)[:-4]
            
            # import_outline=True 会保留原文件自带的目录
            # outline_item 会在总目录中为当前合并的文件创建一个一级标题
            merger.append(pdf, outline_item=bookmark_name)
        
        # 4. 输出文件
        with open(output_filename, "wb") as output_file:
            merger.write(output_file)
        
        merger.close()
        print(f"合并成功！输出文件为: {os.path.abspath(output_filename)}")

        if split_pdf:
            split_large_pdf(output_filename)
        
    except Exception as e:
        print(f"合并过程中出错: {e}")


def get_good_factor(name: str, quant: float):

    test_result = pd.read_csv(os.path.join(name, 'factor_result', 'state_factor_return.csv'))

    test_result.index = test_result['filename'].str.split('/').str[-2].str.split('.').str[0]

    pos_test_result = test_result[test_result['ic_mean'] > 0].sort_values(by=['ic_mean', 'long_ir'], ascending=False)
    neg_test_result = test_result[test_result['ic_mean'] < 0].sort_values(by=['ic_mean', 'short_ir'], ascending=True)

    test_result['choice'] = 0
    test_result.loc[pos_test_result.index[:int(len(pos_test_result) * quant)], 'choice'] = 1
    test_result.loc[neg_test_result.index[:int(len(neg_test_result) * quant)], 'choice'] = -1

    return test_result['choice']


def batch_test(data: pd.DataFrame, name: str, neg_corr: bool = False, smooth_decay: int = 0, split_pdf: bool = False):

    if smooth_decay:
        data = data.groupby("SecuCode").ewm(span=smooth_decay).mean().droplevel(0)
        name = name + f"_EWM{smooth_decay}"

    if neg_corr:
        data = -data
        name = name + "_NEG"

    data = data.swaplevel().sort_index()

    os.makedirs(name, exist_ok=True)
    os.makedirs(os.path.join(name, 'factor_data'), exist_ok=True)
    os.makedirs(os.path.join(name, 'factor_result'), exist_ok=True)

    _format_factor = lambda factor_name: format_factor(data, factor_name, os.path.join(name, 'factor_data'))

    Parallel(n_jobs=4)(
        delayed(_format_factor)(factor_name) for factor_name in tqdm(data.columns, desc="Preparing Data")
    )

    print("Running BackTest")
    # run_shell_cmd(name, format_testor(name))
    expected_folders = os.listdir(os.path.join(name, "factor_data"))
    run_shell_cmd_with_progress(name, format_testor(name), os.path.join(name, 'factor_result'), expected_folders)

    print("Arranging")
    merge_all_pdfs(name, f"{name}.pdf", split_pdf=split_pdf)
    chosen_feature = get_good_factor(name, 0.5)
    result = pd.read_csv(os.path.join(name, "factor_result", "state_factor_return.csv"))
    result.to_csv(f"{name}.csv")
    run_shell_cmd(name, f"rm -r {name}")
    run_shell_cmd(name, f"rm -r {name}batchconfig.json")
    
    return chosen_feature


def neutralize_fast_matrix(sub_df, factor_columns, cap_col='lnCap'):
    """横截面市值中性化核心矩阵运算"""
    x = sub_df[cap_col].values
    X_mat = np.column_stack([x, np.ones(len(x))])
    Y_mat = sub_df[factor_columns].values
    
    coef = np.linalg.lstsq(X_mat, Y_mat, rcond=None)[0]
    res_mat = Y_mat - X_mat.dot(coef)
    
    return pd.DataFrame(res_mat, index=sub_df.index, columns=factor_columns)


def screen_factors(factor_df, 
                   metric='RankIC', 
                   threshold=0.03, 
                   do_neutralize=True,
                   ret_path="Ret5.ftr", 
                   cap_path="Cap.ftr",
                   ret_col='Ret5', 
                   cap_col='lnCap'):
    """
    初步筛选因子 (内部自动加载并处理收益与市值数据)
    
    参数:
    factor_df : pd.DataFrame - 待筛选的因子数据，需为 [SecuCode, DataDate] 的 MultiIndex
    metric    : str - 筛选指标，可选 'RankIC', 'IC', 或 'Both' (Both时默认按RankIC筛选)
    threshold : float - 筛选阈值 (自动取绝对值比较)
    do_neutralize: bool - 是否进行横截面市值中性化
    
    隐藏参数(默认路径和列名，平时调用无需传入):
    ret_path, cap_path : str - 收益率与市值文件路径
    ret_col, cap_col   : str - 收益率与市值的有效列名
    
    返回:
    passed_factors : list - 满足阈值条件的因子名列表
    summary_df     : pd.DataFrame - 所有输入因子的 IC/IR 统计表
    """
    start_time = time.time()
    
    # 1. 内部静默加载收益率和市值数据
    ret_df = pd.read_feather(ret_path).set_index(["SecuCode", "DataDate"]).sort_index()
    cap_df = pd.read_feather(cap_path).set_index(["SecuCode", "DataDate"]).sort_index()
    
    # 2. 数据对齐并剔除空值
    task_df = factor_df.join(ret_df[[ret_col]], how='inner') \
                       .join(cap_df[[cap_col]], how='inner').dropna()
    
    factor_cols = factor_df.columns.tolist()

    print("Data Loaded")
    
    # 3. 市值中性化
    if do_neutralize:
        task_df[factor_cols] = task_df.groupby(level='DataDate', group_keys=False).apply(
            neutralize_fast_matrix, factor_columns=factor_cols, cap_col=cap_col
        )

    print("Neutralization Done")
        
    # 4. 计算指标
    summary_dict = {}
    
    if metric in ['IC', 'Both']:
        daily_ic = task_df.groupby(level='DataDate').apply(
            lambda x: x[factor_cols].corrwith(x[ret_col])
        )
        summary_dict['Mean_IC'] = daily_ic.mean()
        summary_dict['ICIR'] = daily_ic.mean() / daily_ic.std()
        
    if metric in ['RankIC', 'Both']:
        daily_rank_ic = task_df.groupby(level='DataDate').apply(
            lambda x: x[factor_cols].rank().corrwith(x[ret_col].rank())
        )
        summary_dict['Mean_RankIC'] = daily_rank_ic.mean()
        summary_dict['RankICIR'] = daily_rank_ic.mean() / daily_rank_ic.std()
        
    summary_df = pd.DataFrame(summary_dict)
    
    # 5. 根据阈值进行筛选
    target_col = 'Mean_RankIC' if metric in ['RankIC', 'Both'] else 'Mean_IC'
    passed_mask = summary_df[target_col].abs() >= threshold
    passed_factors = summary_df[passed_mask].index.tolist()
    
    print(f"[{metric} 筛选] 阈值: {threshold} | 输入因子数: {len(factor_cols)} | 通过数: {len(passed_factors)} | 耗时: {time.time() - start_time:.2f}s")
    
    return passed_factors, summary_df