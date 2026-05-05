import gc
import torch

def clear_memory():
    """
    Clear memory and cache
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()