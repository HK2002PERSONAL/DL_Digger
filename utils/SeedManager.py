import os
import random
import torch
import numpy as np

def seed_all_non_torch(seed: int = 42) -> None:
    """
    Lock all non-torch random seeds.
    - Python random
    - NumPy random
    - Python hash seed
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_torch_all(seed: int = 42, deterministic: bool = True) -> None:
    """
    Lock all PyTorch random seeds.
    - torch.manual_seed
    - torch.cuda.manual_seed / manual_seed_all
    - deterministic CUDA behavior (optional)
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Optional strict determinism (PyTorch >= 1.8)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass