from typing import Any
import torch.nn as nn

def breakpoint():
    """Temporary fix for breakpoint not working in ipython kernel:
    See:
        https://githubmemory.com/repo/jupyterlab/jupyterlab/issues/10996
        https://github.com/ipython/ipython/issues/13262
    """
    from IPython.core.debugger import set_trace
    return set_trace

def is_frozen(model: nn.Module):
    for p in model.parameters():
        if p.requires_grad:
            return False
    return True


def has_unfrozen_layer(model: nn.Module):
    for p in model.parameters():
        if p.requires_grad:
            return True
    return False


def print_src(pyObj: Any):
    import inspect
    lines = inspect.getsource(pyObj)
    print(lines)