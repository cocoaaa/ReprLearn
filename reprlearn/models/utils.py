import numpy as np
import torch.nn as nn


def inplace_freeze(model: nn.Module):
    """Freezes the modle by turning off its parameter's
    require_grad attributes. In-place operation.
    """
    for p in model.parameters():
        p.requires_grad_(False)


def inplace_unfreeze(model: nn.Module):
    """Freezes the modle by turning off its parameter's
    require_grad attributes. In-place operation.
    """
    for p in model.parameters():
        p.requires_grad_(True)


def compute_ks_for_conv2d(w_in: int, w_out: int, padding: int=1) -> int:
    """Compute the kernel size to use with conv2d when we want
    the output tensor has smaller spatial dimensions, ie.
    w_out < w_in.
    We assume the filter has stride=1.
    Computation is based on the formula
    w_out = floor(w_in - k + 2p) + 1

    We get only positive integer for k only if:
    w_out - w_in < 2p-1
    """
    assert w_out - w_in < 2*padding-1, "No valid kernel size is possible"
    c = w_out - w_in - 2*padding -1
    return -c
