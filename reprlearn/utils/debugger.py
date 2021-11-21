from typing import Any
import torch.nn as nn


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