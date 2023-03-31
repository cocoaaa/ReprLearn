from pathlib import Path
import math
from PIL import Image
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import offsetbox
from matplotlib import colors
import torch
import torchvision
from typing import Tuple, Iterable, Optional, Union, List, Dict
import warnings

from reprlearn.utils.misc import info, has_img_suffix
from reprlearn.data.transforms.functional import unnormalize

from IPython.core.debugger import set_trace as breakpoint
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


def show_n_imgs(img_dir: Path, n_show: int=25, shuffle:bool=False,
               **kwargs) -> plt.Figure:
    """ Show n images in the sample dir. 
    If shuffle is True, show n random images. Otherwise, show n first images.
    
    kwargs is passed to `show_npimgs`:
    - titles: Iterable[Union[str, int]] = None,
    - nrows: int = None,
    - factor=3.0,
    - cmap: str = None,
    - title: Union[str, NoneType] = None,
    - set_axis_off: bool = True,
    """
    if shuffle:
        img_fps = [img_fp for img_fp in img_dir.iterdir() 
                   if img_fp.is_file() and has_img_suffix(img_fp)]
        np.random.shuffle(img_fps) # in-place
            
    else:
        img_fps = []
        for i, img_fp in enumerate(img_dir.iterdir()):
            if len(img_fps) == n_show: 
                break
            if not img_fp.is_file() or not has_img_suffix(img_fp):
                print(f"WARNing: there is a non-valid image file: {img_fp}")
                continue
            img_fps.append(img_fp)
    
    # read each image as np array, show imgs using show_npimgs
    npimgs = [np.array(Image.open(img_fp)) for img_fp in img_fps]
    f, ax = show_npimgs(npimgs, **kwargs)
    return f
            
            
def show_dict_with_colorbar(
    d_arrays: Dict[str,np.ndarray],
    normalizer: colors.Normalize,
    cmap: Optional[colors.Colormap]=None,
    show_colorbar: Optional[bool]=True,
    show_colorbar_every:Optional[int]=None,
    title: Optional[str]=None,
    set_axis_off: Optional[bool]=True,
) -> Tuple[plt.Figure, plt.Axes]:         
    """
    show_color_every (int) : show_colorbar every N cols 
        -e.g. to show one color bar for each row, set it to None,
        and we will set it to the number of columns of the plt.Figure
    """        
    
    arrs = [normalizer(npimg) for npimg in d_arrays.values()]
    dkeys = list(d_arrays.keys())
    n_arrs = len(arrs)
    
    #debug
    # for arr in arrs:
    #     print('min, max: ', arr.min(), arr.max())
#         breakpoint()

    fig, axes = get_fig(len(arrs))
    
    # get num. of rows and colms of the fig
    gs = axes[0].get_gridspec()
    nrows, ncols = gs.nrows, gs.ncols
    if show_colorbar_every is None:
        show_colorbar_every = ncols
        
    for i, ax in enumerate(axes):
        if i < n_arrs:
            pcm = ax.imshow(arrs[i], 
                            # norm=normalizer, 
                            vmin=0., 
                            vmax=1., 
                            cmap=cmap)

            if show_colorbar: #show colorbar next to every axis
                fig.colorbar(pcm, ax=ax, shrink=0.8) #extend='min')

            # alt: show colorbar next to every N columns
            # if show_colorbar and (i+1)%show_colorbar_every == 0:
                # divider = make_axes_locatable(ax)
                # cax = divider.append_axes('right', size='5%', pad=0.05)
                # fig.colorbar(pcm, cax=cax) #extend='min')

            if set_axis_off:
                ax.set_axis_off()
            ax.set_title(dkeys[i])

        else:
            fig.delaxes(ax)
    if title is not None:
        fig.suptitle(title)
    return fig, axes
                   
    

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

def scatterplot_2d(datamtx: np.ndarray, 
                labels: np.ndarray, 
                i2c:Optional[Dict[int, str]]=None, #todo: print the class name rather than index in the legen
                ax: Optional[plt.Axes]=None,
                title:Optional[str]=None,
                figsize: Optional[Tuple[int,int]]=(10,10)) -> plt.Axes:
    """Scatterplot of the datamtx of (N, 2) size.
    
    Args:
    ----
    datamtx: 2dim np array of shape (N, 2). Each column is the first/second coordinate.
    labels: (np.ndarray) of shape (N,) or (N,1). Indicates class/label of each datapt.
    i2c: (Dict) Dictionary mapping class label (int) to a string class-name:
        e.g.: {0:'gan', 1:'vae', 2:'real'}
    ax: (plt.Axes): If not None, make scatterplot on this axes. 
        If None, create a new axis and make scatterplot there.
    title: (str) Title of the scatterplot
    figsize (Tuple[int,int]): size of each axis in the plot in (w,h). 
        Note that the order is not nrows x ncols, but rather widtdh x height.
    
    Returns
    -------
    ax: (plt.Axes) axis with the scatterplot drawn
    """
    assert datamtx.shape[-1] == 2, "Only supports plotting coordinates in 2dim"
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if i2c is not None:
        n_classes = len(i2c.keys())
    else:
        n_classes = len(np.unique(labels))
    
    # scatter plot of data in tsne coords
    sns.scatterplot(
        x=datamtx[:,0],
        y=datamtx[:,1],
        hue=labels, 
        palette=sns.color_palette("hls", n_classes),
        marker='o',
        alpha=0.5,
        legend='full',
        ax=ax,
    )
    
    # if i2c is not None:
    #     #todo: use the classname (str) in the legend, rather than label index (given in labels)
        
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    if title is not None:
        ax.set_title(title)
    
    return ax # or, (ax.get_figure(), ax)



# === Dim reduction/manifold learning algs=== 
# 
def reduce_dim(data: np.ndarray, 
               algo_name: str,
               n_components: int=2,
               n_neighbors: Optional[int]=None, # for isomap, lle
             **kwargs) -> np.ndarray:
    """Given a data in d-dim, apply the specified dim-reduction algorithm 
    to reduce the dimensionality to lower-dim.
    
    Args:
    ----
    data (np.ndarray) : of shape (h,w) = (num_datapts x d), (num_datapts x dim1, dim2, ...)
        - if data.ndim > 2, we will flatten theh data-dimensions to 1 
    algo (str)        : name of the dimension reduction algorithm to apply
    
    Returns:
    -------
    reduced_data (np.ndarray): of shape (h,w) = (num_datapts, r) where r is the 
         reduced dimension.
         
     """
     
    from sklearn import manifold

    if data.ndim > 2:
        data = data.reshape(len(data),-1) 
    algo_name = algo_name.lower()
    if algo_name == 'tsne':
        algo = manifold.TSNE(n_components=n_components,
                    **kwargs)
    elif algo_name == 'iso':
        algo = manifold.Isomap(n_neighbors=n_neighbors, 
                               n_components=n_components,
                               **kwargs)
    elif algo_name == 'lle':
        algo = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, 
                                               n_components=n_components,
                                               **kwargs)
        
    coords = algo.fit_transform(data)
    return coords


def run_and_plot_tsne(data:np.ndarray, 
              labels:np.ndarray, 
              ax:plt.Axes=None,
              title:str=None,
              figsize:Optional[Tuple[int,int]]=None,
              **kwargs_tsne) -> plt.Axes:
    """modified: https://github.com/2-Chae/PyTorch-tSNE/blob/main/main.py
    
    Args
    ----
    - data (np.ndarray): batch of datapoints in shape (N, data-dim1, 2, 3...). 
        We will flatten the data to (N, -1) before running tsne algorithm
    - labels (np.ndarray): labels to use on each datapoint in data. shape: (N,) or (N,1)
        len(labels) must be the same as len(data)
    - out_dir (Path) :  directory to save the tsne plot as pnt
    - seed (int or None): seed to be passed to tsne's random_state argument
    
    Returns
    -------
    ax (plt.Axes): axis with scatterplot of data in tsne coordinates.
        - color-coded according to the labels
    """
    print('generating t-SNE plot...')
    tcoords = reduce_dim(data, algo_name='tsne', **kwargs_tsne)
    ax = scatterplot_2d(tcoords, labels, ax=ax, 
                        title=title,figsize=figsize)

    return ax

    # plt.savefig(out_dir/'tsne.png', bbox_inches='tight')
    # print('Saved tsne plot!')


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
# src: https://github.com/aborgher/Main-useful-functions-for-ML/blob/master/ML/Dimensionality_reduction.ipynb
def plot_2d_coords(data_2d: np.ndarray, 
                   labels: Iterable, 
                   normalize_to_canvas: Optional[bool]=True, #normalize_2d_coords: bool,
                   normalize_dim: Optional[int]=0,
                   ax: Optional[plt.Axes]=None,
                   title: Optional[str]=None,
                   plot_images: Optional[bool]=False,
                   data_images: Optional[np.ndarray]=None,
                #    dist_metric #todo
                   **plt_kwargs):
    """
    Eg:
    ---
    imgs = npimgs (BS, H, w, C)
    labels = batch_labels
    
    coords_2d = dim_reduction(imgs, target_dim=2)
    normalize_2d_coords = False
    normalize_dim = whatever
    
    plot_2d_embedding(data_2d=coords_2d, labels=labels, noramlize_2d_coords=False)
    
    
    
    Args:
    -----
    data (np.ndarray)       : (n_data, dim*)
    lables (List or array)  : (n_data, )
    data_images (np.ndarray): (n_data, img_dim*): array of npimages to be used to show/plot the image at 2d coordinates
    normalize_2d_coords (bool): if True, normalize 2d coordinates 
    normalize_dim (int)     : dimension to normalize over; 0 or 1 (since the coords are in 2dim)
    title (str)
    """    
    n_data = len(data_2d)
    n_labels = len(np.unique(labels))
    if normalize_to_canvas:
        x_min, x_max = np.min(data_2d, normalize_dim), np.max(data_2d, normalize_dim)
        data_2d = (data_2d - x_min) / (x_max - x_min)

    if ax is None:
        _, ax = plt.subplots(nrows=1, ncols=1, **plt_kwargs)
        
    for i in range(n_data):
        # breakpoint()
        ax.text(data_2d[i, 0], data_2d[i, 1], str(labels[i]), 
                color=plt.cm.Set1(labels[i] / n_labels),
                fontdict={'weight': 'bold', 'size': 9}
                ) # todo: set transporancy?
        
        # if (i+1) % 100 == 0:
        #     print(i, end='...')

    if plot_images:
        # plot some images if there is no other images overlapping
        shown_coords = np.array([[1., 1.]])  # just something big
        
        # show image if not overlapping with any other image 
        for i in range(n_data):
            curr_coord = data_2d[i]
            dist = np.sum((curr_coord - shown_coords) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_coords = np.r_[shown_coords, [curr_coord]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox=offsetbox.OffsetImage(data_images[i], cmap=plt.cm.gray_r),
                xy=curr_coord
                )
            ax.add_artist(imagebox)
            
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
                
    # plt.savefig('./plot-test.png')
    # print('saved!')
    # plt.show()
    return ax 