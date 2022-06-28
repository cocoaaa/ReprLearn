import io
import math
from pathlib import Path
from typing import Tuple, Iterable, Optional, Union
import numpy as np
import matplotlib.pyplot as plt


def info(arr, header=None):
    if header is None:
        header = "="*30
    print(header)
    print("shape: ", arr.shape)
    print("dtype: ", arr.dtype)
    print("min, max: ", min(np.ravel(arr)), max(np.ravel(arr)))

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

def save_each_npimg(npimgs: Union[Iterable[np.ndarray], np.ndarray],
                   out_dir: Path,
                    prefix: Union[str, int]='',
                   suffix_start_idx: int=0) -> None:
    
    bs = len(npimgs)
    for i in range(bs):
        idx = suffix_start_idx + i
        fp = out_dir / f'{prefix}_{idx}.png'
        plt.imsave(fp, npimgs[i])
#         print('saved: ', fp)
    

def plt_figure_to_np(fig, dpi=30):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=dpi)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr