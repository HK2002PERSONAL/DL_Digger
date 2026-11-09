import os
import time
import subprocess

from tqdm import tqdm

def run_shell_cmd(name, cmd, mute = True):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    if not mute:
        tqdm.write(f"{name}: {cmd}\noutput: {result.stdout.strip()}") 
    return result.stdout.strip()
    

def thread_checker(pid: int, user: str):
    run_shell_cmd(
        "check pid user", f"""
        if [ "$(ps -o user= -p {pid})" = "${user}" ]; then echo "Yes, it is YOUR thread."; else echo "No, it belongs to: $(ps -o user= -p {pid})"; fi
        """, False)


def run_shell_cmd_with_file_addition_check(
    name: str, 
    cmd: str, 
    target_path: os.PathLike, 
    expected_folders: os.PathLike, 
    appendix_num: int = 5, 
    check_interval: int = 10
):
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
            
            # 每 check_interval 秒检查一次
            time.sleep(check_interval)
            
        # 4. 进程结束后，确保进度条满格（处理最后一次扫描到结束之间的间隙）
        if last_count < total:
            pbar.update(total - last_count)

    # 获取最终输出
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        tqdm.write(f"✅ {name} 完成: {stdout}")
    else:
        tqdm.write(f"❌ {name} 失败: {stderr}")