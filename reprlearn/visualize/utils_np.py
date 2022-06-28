from pathlib import Path
import math
from PIL import Image
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Iterable, Optional, Union, List
import warnings


"""
TODO:
- Eventually I want to separate out get_fig and show_npimgs to a module that doesn't import torch or torchvision
    - because, these functions don't require torch library, and allows users of simply numpy and matplotlib to visualize 
    a list of numpy arrays (eg. images)
    - In that module, e.g. named as "src.utils.nparr", include any function that deals with  numpy arrays -- vs. not tensors
        - e.g. src.utils.misc.info
        

"""
def has_img_suffix(fp: Path, valid_suffixes:List[str]=['.png', '.jpeg', '.jpg']):
    return fp.suffix.lower() in valid_suffixes


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


def set_plot_lims(ax: plt.Axes, xlim: Tuple[float,float], ylim: Tuple[float, float]):
    ax.set_xlim(xmin=xlim[0], xmax=xlim[1])
    ax.set_ylim(ymin=ylim[0], ymax=ylim[1])
    return ax


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