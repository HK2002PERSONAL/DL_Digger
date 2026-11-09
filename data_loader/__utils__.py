import os
from typing import List

# ==============================
# 全局路径配置
# ==============================
SOURCE_PATH = os.path.join('/data', 'cephfs', 'minute')
TARGET_PATH = 'data'

# ==============================
# 工具函数（工程化改造点 #2 / #3）
# ==============================
def _list_days_from_dir(base_dir: str, suffix_len: int = 8) -> List[str]:
    """
    从目录文件名中提取交易日字符串并排序。

    约定：文件名（不含路径）前8位是 YYYYMMDD，例如 20200104.feather。
    """
    daylist = os.listdir(base_dir)
    return sorted([name[:suffix_len] for name in daylist])


def _get_lookback_days(day: str, daylist: List[str], lookback: int) -> List[str]:
    """
    从 daylist 中取 <= day 的最近 lookback 个交易日。
    """
    return [d for d in daylist if d <= day][-lookback:]


def _ensure_dir(path: str) -> None:
    """
    若目录不存在则创建。
    """
    os.makedirs(path, exist_ok=True)
