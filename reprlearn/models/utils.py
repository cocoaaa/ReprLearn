import numpy as np

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
