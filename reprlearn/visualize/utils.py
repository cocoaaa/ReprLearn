from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import pytorch_lightning as pl
from typing import Tuple, Iterable, Optional, Union, List
import warnings

from src.utils.misc import info
from src.data.transforms.functional import unnormalize

"""
TODO:
- Eventually I want to separate out get_fig and show_npimgs to a module that doesn't import torch or torchvision
    - because, these functions don't require torch library, and allows users of simply numpy and matplotlib to visualize 
    a list of numpy arrays (eg. images)
    - In that module, e.g. named as "src.utils.nparr", include any function that deals with  numpy arrays -- vs. not tensors
        - e.g. src.utils.misc.info
        

"""
def get_fig(n_total: int, nrows: int=None, factor=3.0) -> Tuple[plt.Figure, plt.Axes]:
    """Create a tuple of plt.Figure and plt.Axes with total number of subplots `n_total` with `nrows` number of rows.
    By default, nrows and ncols are sqrt of n_total.

    :param n_total: total number of subplots
    :param nrows: number of rows in this Figure
    :param factor: scaling factor that is multipled to both to the row and column sizes
    :return: Tuple[Figure, flatten list of Axes]
    """
    if nrows is None:
        nrows = math.ceil(n_total ** .5)

    ncols = math.ceil(n_total / nrows)
    f, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(factor * ncols, factor * nrows))
    axes = axes.flatten()
    return f, axes


def show_npimgs(npimgs: Iterable[np.ndarray], *,
                titles: Iterable[Union[str, int]]=None,
                nrows: int=None,
                factor=3.0,
                cmap:str = None,
                title: Optional[str] = None,
                set_axis_off: bool=True) -> Tuple[plt.Figure, plt.Axes]:
    n_imgs = len(npimgs)
    f, axes = get_fig(n_imgs, nrows=nrows, factor=factor)

    for i, ax in enumerate(axes):
        if i < n_imgs:
            ax.imshow(npimgs[i], cmap=cmap)

            if titles is not None:
                ax.set_title(titles[i])
            if set_axis_off:
                ax.set_axis_off()
        else:
            f.delaxes(ax)
    if title is not None:
        f.suptitle(title)
    return f, axes


def show_timg(timg: torch.Tensor,
              title=None,
              fig_title=None,
              subplots_kw=None,
              imshow_kw=None,
              axis_off=True,
              save_path: Union[str, Path]=None) -> plt.Axes:
    """

    :param timg: 3dim tensor in order (c,h,w)
    :param subplots_kw:
        eg. figsize=(width, height)
    :param imshow_kw:
        eg. cmap, norm, interpolation, alpha
    :return:
    """
    # Set defaults
    if subplots_kw is None:
        subplots_kw = {}
    if imshow_kw is None:
        imshow_kw = {}

    npimg = timg.numpy()

    f, ax = plt.subplots(**subplots_kw)
    ax.imshow(np.transpose(npimg, (1, 2, 0)),
               interpolation='nearest',
               **imshow_kw)
    if axis_off:
        ax.set_axis_off()

    if title is not None:
        ax.set_title(title)
    if fig_title is not None:
        f.suptitle(title)

    if save_path is not None:
        f.savefig(save_path)

    return ax


def show_timgs(timgs: Iterable[torch.Tensor], order='chw', **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Assumes timgs has a shape order of (bs, nchannels, h, w) if order=='chw'.
    If not, assumes timgs has a shape order of (bs, h, w, nchannels)

    **kwargs will be passed into `show_npimgs` function
    - titles: a list of titles for axes; must be the same length as the number of npimgs
    - nrows
    - factor
    - cmap (str): eg. "gray"
    - title (str): suptitle of the main figure
    - titles (List[str]): a list of axis titles
    """
    try:
        npimgs = timgs.numpy()
    except AttributeError:
        npimgs = np.array([t.numpy() for t in timgs]) #todo check

    if order == 'chw':
        npimgs = npimgs.transpose(0, -2, -1, 1)

    f, axes = show_npimgs(npimgs, **kwargs)

    return f, axes


def show_batch(dm, #: BaseDataModule,
               mode: str='train',
               n_show: int = 16,
               show_unnormalized: bool = True,
               **kwargs):
    """
    Show a batch of train data in the datamodule.
    -kwargs will be passed into `show_timgs` function
        - titles: a list of titles for axes; must be the same length as the number of npimgs
        - nrows
        - factor
        - cmap (str): eg. "gray"
        - title (for the main figure's suptitle)
    """
    # with torch.no_grad() and
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="matplotlib*")

        dl = getattr(dm, f"{mode}_dataloader")()
        x, y = next(iter(dl))

        if show_unnormalized:
            train_mean, train_std = dm.train_mean, dm.train_std
            # Undo normalization
            x_unnormed = unnormalize(x, train_mean, train_std)
            info(x_unnormed, "unnormalized x")
            show_timgs(x_unnormed[:n_show], **kwargs)
        else:
            info(x, "batch x")
            show_timgs(x[:n_show], **kwargs)


def make_grid_from_tensors(tensors: List[torch.Tensor], dim: int=-1) -> torch.Tensor:
    """
    Make a single tensor of shape (C, gridH, gridW) by concatenating the tensors in the argument.
    Assumes all tensors have the same number of channels.
    Args:
    - dim (int):
        - use -1 to put the tensors side-by-side
        - use -2 to put them one below another

    Example:
    grid = make_grid_from_tensors([grid_input, grid_recon], dim=-1)
    tb_writer.add_image("input-recon", grid, global_step=0)
    """
    grids = [torchvision.utils.make_grid(t) for t in tensors]  # each element has size, eg.(C, gridh, gridw)
    combined = torch.cat(grids, dim=dim)
    return combined