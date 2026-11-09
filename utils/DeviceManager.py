import os

from .BackTester.utils.run_cmd import run_shell_cmd
from .Logger import smart_log # pylint:disable=relative-beyond-top-level

os.environ['CUDA_VISIBLE_DEVICES'] = run_shell_cmd('device authority check', 'echo $CUDA_VISIBLE_DEVICES')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch

def get_device(
    device_id=0,               # GPU ID, default 0
    max_memory_usage=0.1,      # Maximum memory usage ratio (30%)
    verbose=True               # Whether to print detailed info
):
    """
    Get available device. Returns CPU if CUDA unavailable, raises error if GPU is busy.
    
    Args:
        device_id: GPU ID (0, 1, 2, ...)
        max_memory_usage: Maximum allowed memory usage ratio (0-1), default 0.3 (30%)
        verbose: Whether to print detailed information
    
    Returns:
        torch.device: Available device (cuda:X or cpu)
    
    Raises:
        RuntimeError: If CUDA is available but GPU is occupied
    
    Examples:
        >>> # Basic usage
        >>> device = get_device(device_id=0)
        >>> model = Model().to(device)
        
        >>> # Specify GPU and threshold
        >>> device = get_device(device_id=1, max_memory_usage=0.5)
        
        >>> # Silent mode
        >>> device = get_device(device_id=0, verbose=False)
    """
    
    # ========================================
    # 1. Check if CUDA is available
    # ========================================
    if not torch.cuda.is_available():
        if verbose:
            smart_log("⚠️ CUDA not available, using CPU")
        return torch.device('cpu')
    
    # ========================================
    # 2. Check if GPU ID is valid
    # ========================================
    _gpus = run_shell_cmd('device authority check', 'echo $CUDA_VISIBLE_DEVICES').split(',')
    
    if str(device_id) not in _gpus:
        msg = (
            f"GPU {device_id} does not exist or without authority!\n"
            f"  Available GPUs: {_gpus}\n"
        )
        raise RuntimeError(msg)

    virtual_device_id = _gpus.index(str(device_id))
    
    # ========================================
    # 3. Check GPU memory usage
    # ========================================
    torch.cuda.set_device(virtual_device_id)
    free_memory, total_memory = torch.cuda.mem_get_info(virtual_device_id)
    used_memory = total_memory - free_memory
    usage_ratio = used_memory / total_memory
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(virtual_device_id)
    
    # ========================================
    # 4. Print detailed info (if verbose=True)
    # ========================================
    if verbose:
        smart_log("="*60)
        smart_log(f"GPU {device_id} Status Check")
        smart_log("="*60)
        smart_log(f"Name: {gpu_name}")
        smart_log(f"Total Memory: {total_memory / 1e9:.2f} GB")
        smart_log(f"Used Memory: {used_memory / 1e9:.2f} GB ({usage_ratio * 100:.1f}%)")
        smart_log(f"Free Memory: {free_memory / 1e9:.2f} GB")
        smart_log(f"Usage Threshold: {max_memory_usage * 100:.1f}%")
        smart_log("="*60)
    
    # ========================================
    # 5. Check if exceeds threshold
    # ========================================
    if usage_ratio > max_memory_usage:
        msg = (
            f"GPU {device_id} memory usage is too high!\n"
            f"  GPU Name: {gpu_name}\n"
            f"  Current Usage: {usage_ratio*100:.1f}%\n"
            f"  Threshold: {max_memory_usage*100:.1f}%\n"
            f"  Used Memory: {used_memory/1e9:.2f} GB / {total_memory/1e9:.2f} GB\n"
            f"\nSuggestions:\n"
            f"  1. Use another GPU (modify device_id parameter)\n"
            f"  2. Wait for current tasks to complete\n"
            f"  3. Increase threshold (modify max_memory_usage parameter)\n"
            f"  4. Use nvidia-smi to check detailed status"
        )
        raise RuntimeError(msg)
    
    # ========================================
    # 6. GPU available, return device
    # ========================================
    device = torch.device(f'cuda:{virtual_device_id}')
    
    if verbose:
        smart_log(f"✓ GPU {device_id} is available (seen as {device})")
        print()
    
    return device


def get_free_device(
    max_memory_usage=0.1,      # Maximum memory usage ratio (10%)
    verbose=True               # Whether to print detailed info
):
    """
    Get the first free GPU device. Iterates through all available GPUs and returns
    the first one with memory usage below the threshold.
    
    Args:
        max_memory_usage: Maximum allowed memory usage ratio (0-1), default 0.1 (10%)
        verbose: Whether to print detailed information
    
    Returns:
        torch.device: Available device (cuda:X or cpu)
    
    Raises:
        RuntimeError: If no free GPU is available
    
    Examples:
        >>> # Basic usage
        >>> device = get_free_device()
        >>> model = Model().to(device)
        
        >>> # With higher threshold
        >>> device = get_free_device(max_memory_usage=0.3)
        
        >>> # Silent mode
        >>> device = get_free_device(verbose=False)
    """
    
    # ========================================
    # 1. Check if CUDA is available
    # ========================================
    if not torch.cuda.is_available():
        if verbose:
            smart_log("⚠️ CUDA not available, using CPU")
        return torch.device('cpu')
    
    # ========================================
    # 2. Get available GPU list
    # ========================================
    _gpus = run_shell_cmd('device authority check', 'echo $CUDA_VISIBLE_DEVICES').split(',')
    _gpus = [g.strip() for g in _gpus if g.strip()]  # Remove empty entries
    
    if not _gpus:
        raise RuntimeError("No GPUs found in CUDA_VISIBLE_DEVICES!")
    
    if verbose:
        smart_log("="*60)
        smart_log("Scanning GPUs for free device...")
        smart_log(f"Available GPUs: {_gpus}")
        smart_log(f"Memory threshold: {max_memory_usage * 100:.1f}%")
        smart_log("="*60)
    
    # ========================================
    # 3. Iterate through GPUs to find a free one
    # ========================================
    for virtual_device_id, device_id in enumerate(_gpus):
        torch.cuda.set_device(virtual_device_id)
        free_memory, total_memory = torch.cuda.mem_get_info(virtual_device_id)
        used_memory = total_memory - free_memory
        usage_ratio = used_memory / total_memory
        
        gpu_name = torch.cuda.get_device_name(virtual_device_id)
        
        if verbose:
            smart_log(
                f"GPU {device_id} (cuda:{virtual_device_id}): "
                f"{gpu_name} - "
                f"Usage: {usage_ratio * 100:.1f}% "
                f"({used_memory/1e9:.2f} GB / {total_memory/1e9:.2f} GB)"
            )
        
        # Check if this GPU is free
        if usage_ratio <= max_memory_usage:
            device = torch.device(f'cuda:{virtual_device_id}')
            if verbose:
                smart_log("="*60)
                smart_log(f"✓ GPU {device_id} is free (seen as {device})")
                print()
            return device
    
    # ========================================
    # 4. No free GPU found
    # ========================================
    msg = (
        "No free GPU available!\n"
       f"  Scanned GPUs: {_gpus}\n"
       f"  Memory threshold: {max_memory_usage * 100:.1f}%\n"
        "\nSuggestions:\n"
        "  1. Wait for current tasks to complete\n"
        "  2. Increase threshold (modify max_memory_usage parameter)\n"
        "  3. Use nvidia-smi to check detailed status"
    )
    raise RuntimeError(msg)
