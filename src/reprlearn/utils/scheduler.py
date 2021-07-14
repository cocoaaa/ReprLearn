import numpy as np
from torch.utils.data import DataLoader

def n_iters_per_epoch(dl:DataLoader) -> int:
    """Given a dataloader
    - batch_size,
    - total num. of data in the dataset
    Compute the total number of batch/iterations per epoch.
    """
    n_iters = len(dl.dataset)//dl.batch_size
    if not dl.drop_last:
        n_iters += 1
    return n_iters

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return iter(L)