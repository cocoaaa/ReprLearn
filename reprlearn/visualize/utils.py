from pathlib import Path
import math
from PIL import Image
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import pytorch_lightning as pl
from typing import Tuple, Iterable, Optional, Union, List
import warnings

from reprlearn.utils.misc import info, has_img_suffix
from reprlearn.data.transforms.functional import unnormalize

"""
TODO:
- Eventually I want to separate out get_fig and show_npimgs to a module that doesn't import torch or torchvision
    - because, these functions don't require torch library, and allows users of simply numpy and matplotlib to visualize 
    a list of numpy arrays (eg. images)
    - In that module, e.g. named as "src.utils.nparr", include any function that deals with  numpy arrays -- vs. not tensors
        - e.g. src.utils.misc.info
        

"""

def get_ith_npimg(img_dir: Path, ind: int, show:bool=True) -> np.ndarray:
    """
    Return the `ind`th image in `img_dir` assuming the imag_dir is strauctured as:
    <img_dir>
        |- <fn1>.png
        |- <fn2>.png
        |- ...

    :param img_dir: directory that contains images
    :param ind: `ind`th image will be returned
    :param show: bool to show the loaded image
    :return:
    """
    i = 0
    for img_fp in img_dir.iterdir():
        if img_fp.is_file() and has_img_suffix(img_fp):
            if i == ind:
                #                 print('found')
                pil_img = Image.open(img_fp)
                npimg = np.array(pil_img)
                if show:
                    plt.title(i)
                    plt.imshow(npimg)
                    plt.axis('off')
                return npimg
            else:
                i += 1


get_first_npimg = partial(get_ith_npimg, ind=0)


def normalize_timg_ip(tensor: torch.Tensor,
                      value_range: Optional[Tuple[int, int]] = None) -> None:
    """ In-place normalization of a single tensor
    Modified from src: torchvision.utils.make_grid"""
    tensor = tensor.clone()  # avoid modifying tensor in-place
    if value_range is not None:
        assert isinstance(value_range, tuple), \
            "value_range has to be a tuple (min, max) if specified. min and max are numbers"

    def norm_ip(img, low, high):
        "in-place operation"
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    def norm_range(t, value_range):
        if value_range is not None:
            norm_ip(t, value_range[0], value_range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))
    norm_range(tensor, value_range)
    return tensor


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
    - dm : (pl.DataModule) must have `unpack` method which unpakcs a batch from the dataloader
        to x,y
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
        batch = next(iter(dl)) # Dict[str,Any] as the contract on my impl. of Dataset class (__getitem__ returns Dict[str,Any])
        x = batch['x']

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