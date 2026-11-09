import os
import sys
import inspect
import traceback
import threading
import datetime

from tqdm import tqdm

# 定義全局鎖
_log_lock = threading.Lock()

# 定义全局日志文件路径变量
_global_log_file = "logging.log"  # 默认日志文件路径

def make_log_file():
    """
    设置全局日志文件路径
    """
    stack = inspect.stack()
    main_file = os.path.basename(stack[-1].filename)
    global _global_log_file
    _global_log_file = os.path.join('_logs', f'{main_file[:-3]}.log')
    if not os.path.exists(_global_log_file):
        os.makedirs(os.path.dirname(_global_log_file), exist_ok=True)  # 确保目录存在
        with open(_global_log_file, "w", encoding="utf-8") as f: pass # 创建空文件

def smart_log(msg, tqdm_mode=False):
    """
    通用日誌函數：
    1. 輸出到 IO (控制台)
    2. 帶日期寫入文件
    3. 線程安全
    """
    stack = inspect.stack()
    main_file = os.path.basename(stack[-1].filename)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_msg = f"[{timestamp}][running {main_file}] {msg}"
    
    # 1. 輸出到 IO
    # 注意：使用 flush=True 確保程序崩潰前信息能打印出來
    if tqdm_mode:
        tqdm.write(msg)
    else:
        print(msg, flush=True) 

    global _global_log_file
    
    # 2. 帶鎖寫入文件
    with _log_lock:
        try:
            with open(_global_log_file, "a", encoding="utf-8") as f:
                f.write(file_msg + "\n")
        except Exception as e:
            # 如果連寫日誌都失敗，直接打印到控制台
            print(f"CRITICAL: Failed to write to log file: {e}", file=sys.stderr)

def handle_exception(exc_type, exc_value, exc_traceback):
    """
    系統異常攔截器：
    捕獲所有未處理的報錯，並通過 smart_log 輸出
    """
    # 忽略手動中斷 (Ctrl+C)
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # 獲取完整報錯信息
    error_details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    
    # 使用 smart_log 同時發送到 IO 和 文件
    log_header = "\n" + "!"*20 + " PROGRAM CRASHED " + "!"*20
    smart_log(f"{log_header}\n{error_details}{'!'*57}\n")

# 激活全局異常監聽
sys.excepthook = handle_exception