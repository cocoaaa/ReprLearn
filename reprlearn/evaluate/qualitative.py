from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import linalg as LA

import torchvision
from torchvision.utils import make_grid

import pytorch_lightning as pl

from src.data.transforms.transforms import LinearRescaler
from src.data.transforms.functional import unnormalize

from src.models.plmodules import BiVAE
from src.visualize.utils import show_timg, show_timgs, show_batch
from src.utils.misc import now2str, info, get_next_version_path


def show_recon(model: pl.LightningModule,
               dm: pl.LightningDataModule,
               tb_writer: SummaryWriter = None,
               global_step: int = 0,
               unnorm: bool = True,
               to_show: bool = True,
               verbose: bool = False):
    model.eval()
    cmap = 'gray' if dm.size()[0] == 1 else None
    train_mean, train_std = dm.train_mean, dm.train_std
    dl_modes = ['test'] if hasattr(dm, 'test_ds') else ['train', 'val']
    with torch.no_grad():
        for mode in dl_modes:
            dl = getattr(dm, f"{mode}_dataloader")()
            batch = next(iter(dl))

            if isinstance(batch, dict):
                x = batch['img']
                #             label_c = batch['digit']  # digit/content label (int) -- currently not used
                #             label_s = batch['color']
            else:
                x, _ = batch

            x = x.to(model.device)
            x_recon = model.reconstruct(x)

            # Move to cpu for visualization
            x = x.cpu()
            x_recon = x_recon.cpu()

            if verbose:
                info(x, f"{mode}_x")
                info(x_recon, f"{mode}_x_recon")

            if unnorm:
                x_unnormed = unnormalize(x, train_mean, train_std)
                x_recon_unnormed = unnormalize(x_recon, train_mean, train_std)
                if verbose:
                    print("===After unnormalize===")
                    info(x_unnormed, f"{mode}_x_unnormed")
                    info(x_recon_unnormed, f"{mode}_x_recon_unnormed")

            if to_show:
                _x = x_unnormed if unnorm else x
                _x_recon = x_recon_unnormed if unnorm else x_recon
                show_timgs(_x, title=f"Input: {mode}", cmap=cmap)
                #                 show_timgs(_x_recon, title=f"Recon: {mode}", cmap=cmap)
                show_timgs(LinearRescaler()(_x_recon), title=f"Recon(linearized): {mode}", cmap=cmap)

            # Log input-recon grid to TB
            if tb_writer is not None:
                input_grid = make_grid(x_unnormed)  # (C, gridh, gridw)
                recon_grid = make_grid(x_recon_unnormed)  # (C, gridh, gridw)
                normed_recon_grid = make_grid(LinearRescaler()(x_recon_unnormed))

                grid = torch.cat([input_grid, normed_recon_grid], dim=-1)  # inputs | recons
                tb_writer.add_image(f"{mode}/recons", grid, global_step=global_step)


# ------------------------------------------------------------------------
# Latent Space
# ------------------------------------------------------------------------
def get_class_reps(dl: DataLoader) -> Dict[Union[str,int], torch.Tensor]:
    class_reps = {}
    for i in range(len(dl.dataset)):
        batch = dl.dataset[i]
        x = batch['img']
        label_c = batch['digit']  # digit/content label (int) -- currently not used
        label_s = batch['color']

        if len(class_reps) >= 10:
            break
        if isinstance(label_c, torch.Tensor):
            label_c = label_c.item()
        label_c = str(label_c)
        if label_c in class_reps:
            continue
        class_reps[label_c] = x
    return class_reps

def evaluate_transfers(model: BiVAE,
                    constant_code: str, #"c" or "s"
                    class_reps: Dict[str, torch.Tensor],
                    log_dir: Path,
                    train_mean: List, # to unnormalize output tensor
                    train_std: List, # to unnormalize
                    linearlize: bool=True):
    """

    :param model: Trained BiVAE model
    :param class_reps: a dictionary of string class_id <-> a single 3dim Tensor (C,H,W)
    :param log_dir: Path to the model.logger's log_dir (Eg. '..../exp_name/version7')
    :param train_mean: Original datamodule's training set's mean
    :param train_std:  Oiriginal datamodule's training set std
    :param linearlize: (bool). If true, linearlize the output image to range [0,1] for better viz. contrast
    :return: a plt.Figure of all transfer results put together via torchvision.make_grid
    """
    assert constant_code in ['c', 's'], "constant_code must be 'c' or 's' (for content, style, respectively"
    model.eval()
    ids = sorted(class_reps.keys())
    grids = {}
    for i, id_a in enumerate(ids):

        grids[id_a] = []
        for j, id_b in enumerate(ids):

            img_a = class_reps[id_a]
            img_b = class_reps[id_b]
            img_pair = torch.stack([img_a, img_b], dim=0)
            unnormed_img_pair = unnormalize(img_pair, train_mean, train_std)

            with torch.no_grad():
                dict_qparams = model.encode(img_pair)
                dict_z = model.rsample(dict_qparams)

                if constant_code == 'c':
                    # Fix content to c[0]
                    content = dict_z["c"][[0]]
                    style = dict_z["s"][[1]]
                elif constant_code == 's':
                    # Fix style to s[0]
                    content = dict_z["c"][[1]]
                    style = dict_z["s"][[0]]
                test_dict_z = {"c": content, "s": style}

                # Reconstruct
                z = model.combine_content_style(test_dict_z)
                recons = model.decode(z)

                # Optional: for better viz, unnormalize or/and linearlize
                unnormed_recons = unnormalize(recons, train_mean, train_std)
                if linearlize:
                    img_pair = LinearRescaler()(img_pair) #todo
                    unnormed_recons = LinearRescaler()(unnormed_recons)

                grid = make_grid(
                    torch.cat([unnormed_img_pair, unnormed_recons], dim=0)
                )  # (3, gridh, gridw)
                grids[id_a].append(grid)
            # grids[id_a] is a list of each row result, i.e length = 10


    # Save each content's transferred image as:
    # log_dir/content_transfers/{next_version}/"content_transfers_{content-id}.png"
    constant_type = "content" if constant_code == "c" else "style"
    save_dir = get_next_version_path(log_dir, name=f'{constant_type}_transfers')

    if not save_dir.exists():
        save_dir.mkdir()
        print("Created: ", save_dir)

    all_recons = []
    for id_a, recons in grids.items():
        recons = torch.cat(recons, dim=1) # recons is a 3dim tensor (3, gridh, gridw) for each id_a
        all_recons.append(recons)

        # Show and save this id's transfers
        save_path = save_dir/f'{constant_type}_transfers_{id_a}.png'
        show_timg(recons,
                  title=id_a,
                  save_path=save_path,
                 )
    # Put all per-id transfer results into a single grid
    return all_recons# make_grid(all_recons, nrow=10, padding=30)


def save_content_transfers(model: BiVAE, *args, **kwargs):
    """

    :param model: Trained BiVAE model
    :param class_reps: a dictionary of string class_id <-> a single 3dim Tensor (C,H,W)
    :param log_dir: Path to the model.logger's log_dir (Eg. '..../exp_name/version7')
    :param train_mean: Original datamodule's training set's mean
    :param train_std:  Oiriginal datamodule's training set std
    :param linearlize: (bool). If true, linearlize the output image to range [0,1] for better viz. contrast
    :return:
    """
    return evaluate_transfers(model, constant_code='c', *args, **kwargs)

def save_style_transfers(model: BiVAE, *args, **kwargs):
    """

    :param model: Trained BiVAE model
    :param class_reps: a dictionary of string class_id <-> a single 3dim Tensor (C,H,W)
    :param log_dir: Path to the model.logger's log_dir (Eg. '..../exp_name/version7')
    :param train_mean: Original datamodule's training set's mean
    :param train_std:  Oiriginal datamodule's training set std
    :param linearlize: (bool). If true, linearlize the output image to range [0,1] for better viz. contrast
    :return:
    """
    return evaluate_transfers(model, constant_code='s', *args, **kwargs)

def run_both_transfers(model: BiVAE, *args, **kwargs):
    """
    Run both content-transfer and style-transfer on the each pair of the content-representative tensor images
    :param model: Trained BiVAE model
    :param class_reps: a dictionary of string class_id <-> a single 3dim Tensor (C,H,W)
    :param log_dir: Path to the model.logger's log_dir (Eg. '..../exp_name/version7')
    :param train_mean: Original datamodule's training set's mean
    :param train_std:  Oiriginal datamodule's training set std
    :param linearlize: (bool). If true, linearlize the output image to range [0,1] for better viz. contrast
    :return:
    """
    return (save_content_transfers(model, *args, **kwargs), save_style_transfers(model, *args, **kwargs))


def compute_avg_codes(
        model: pl.LightningModule,
        dm: pl.LightningDataModule,
        batch_size: Optional[int] = None,
        mode: str='train'
) -> Dict[str,torch.Tensor]:
    """Given a model and datamodule:
    - Get a batch of input -> encode -> get a batch of mu_qc, mu_qs -> get samples of content and style codes
    - Compute the dimension-wise mean/min/max of mu_qc, the mean/min/max of mu_qs,
    mean of sampled content codes, mean of sampled style codes.

    Returns
    -------
     the computed mean/min/max results in a dictionary.

    :param model:
    :param dm:
    :param batch_size:
    :param mode:
    :return:
    """

    n_styles = dm.n_styles
    n_contents = dm.n_contents
    ds = getattr(dm, f"{mode}_ds")
    batch_size = batch_size or len(ds)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=16, pin_memory=True, shuffle=True)

    with torch.no_grad():
        batch = next(iter(dl))
        # x = batch['img'] # todo: unpack a sample from dataset object using DS.unpack
        # label_c = batch['digit']
        # label_s = batch['color']
        x, label_c, label_s = dm.unpack(batch)

        dict_qparams = model.encode(x)
        mu_qc = dict_qparams['mu_qc']
        mu_qs = dict_qparams['mu_qs']

        dict_z = model.rsample(dict_qparams)
        c = dict_z['c']
        s = dict_z['s']
        z = model.combine_content_style(dict_z)
        # mean norm of each code
        norm_c = LA.norm(c, dim=-1)
        norm_s = LA.norm(s, dim=-1)
        print("Avg. norm of content content: ", norm_c.mean())
        print("Avg. norm of style content: ", norm_s.mean())

        content_avgs = {}
        mu_qc_avgs = {}
        mu_qc_mins = {}
        mu_qc_maxs = {}
        for content_id in range(n_contents):
            this_mu_qc = mu_qc[label_c == content_id]
            mu_qc_avgs[content_id] = this_mu_qc.mean(dim=0)
            mu_qc_mins[content_id] = this_mu_qc.min(dim=0).values
            mu_qc_maxs[content_id] = this_mu_qc.max(dim=0).values
            # mean of samples
            mean_c = c[label_c == content_id].mean(dim=0)
            content_avgs[content_id] = mean_c

        style_avgs = {}
        mu_qs_avgs = {}
        mu_qs_mins = {}
        mu_qs_maxs = {}
        for style_id in range(n_styles):
            this_mu_qs = mu_qs[label_s == style_id]
            mu_qs_avgs[style_id] = this_mu_qs.mean(dim=0)
            mu_qs_mins[style_id] = mu_qs[label_s == style_id].min(dim=0).values
            mu_qs_maxs[style_id] = mu_qs[label_s == style_id].max(dim=0).values
            # mean of samples
            mean_s = s[label_s == style_id].mean(dim=0)
            style_avgs[style_id] = mean_s
    return {"content_avgs":content_avgs,
            "mu_qc_avgs": mu_qc_avgs,
            "mu_qc_mins": mu_qc_mins,
            "mu_qc_maxs": mu_qc_maxs,
            "style_avgs": style_avgs,
            "mu_qs_avgs": mu_qs_avgs,
            "mu_qs_mins": mu_qs_mins,
            "mu_qs_maxs": mu_qs_maxs}


def get_traversals(
        vec: torch.Tensor,
        dim_i: int,
        min_value: float,
        max_value: float,
        n_samples: int
) -> torch.Tensor:
    """Given a single 1dim vector, get a batch of 1dim vectors by traversing
    `dim_i`th dimension of `vec` linearly, from `min_value` to `max_value`

    Returns
    -------
    traversals : torch.Tensor (1 more dims than `vec`)
        a batch of 1dim vectors, generated by linearly traversing `dim_i`th
        dimension of `vec`
    """
    new_values = torch.linspace(min_value, max_value, n_samples)
    traversals = torch.zeros((n_samples, len(vec)))

    for i, new_value in enumerate(new_values):
        new_vec = vec.clone()
        new_vec[dim_i] = new_value
        traversals[i] = new_vec
    return traversals


def run_content_traversal(
        model: pl.LightningModule,
        content_code: torch.Tensor,
        style_code: torch.Tensor,
        traversal_start: Union[float, Iterable[float]],
        traversal_end: Union[float, Iterable[float]],
        n_traversals: int,
        show: bool = True,
        title: str = '',
        to_save: bool = True,
        out_path: Optional[Path] = None,
        verbose: bool=False,
) -> torch.Tensor:
    """Given a fixed style code, traverse the content code each dimension  $j$
    independently, from `traversal_start[j]` to `traversal_end[j]` at
    `n_traversals` steps.

    Parameters
    ----------
        model : pl.LightningModule
        content_code : torch.Tensor; shape (dim_content, )
        style_code : torch.Tensor; shape (dim_style,)
        traversal_start : Iterable; length == dim_content
            a vector of floats that indicate the starting point of the traversal
            for each dimsion of the content code
        traversal_end : Iterable; length == dim_content
            a vector of floats that indicate the ending point of the traversal
            for each dimsion of the content code
        n_traversals : int
            how many traversal steps per dimension
        show : bool
            True to show the result in a grid where $j$th col is the content dim $j$,
            and the $i$th row shows the $i$th step in that direction.
        title : str

    Returns
    -------
        torch.Tensor; shape (`n_traversals`, dim_content, *dim_input_x)
            a batch of reconstructions from each dimension's traversals.
            Eg: output[n][j] contains a (C,H,W) image reconstructed by a $n$th
            step at content_code's dim jth direction with the fixed style code.
    """
    is_training = model.training
    model.eval()
    content_dim = content_code.shape[-1]
    try:
        traversal_start[0]
    except TypeError:
        traversal_start = torch.zeros(content_dim).fill_(traversal_start)
    try:
        traversal_end[0]
    except TypeError:
        traversal_end = torch.zeros(content_dim).fill_(traversal_end)
    with torch.no_grad():
        # Traverse for each dim
        grids = []  # k,v = dim_i, batch of recons while traversing in dim_i direction (n_traversals, *dim_x)
        for dim_i in range(content_dim):
            min_dim_i = traversal_start[dim_i]
            max_dim_i = traversal_end[dim_i]
            if verbose:
                print(min_dim_i, max_dim_i)
            c_traversals = get_traversals(content_code, dim_i, min_dim_i, max_dim_i, n_traversals)

            # Pass to the decoder
            dict_z = {
                "c": c_traversals,
                "s": style_code.repeat((n_traversals, 1))
            }
            z = model.combine_content_style(dict_z)
            recons = model.decode(z)

            grid = torchvision.utils.make_grid(recons, nrow=1)  # Caveat: nrow is num of colms!
            grids.append(grid)
        grids = torch.cat(grids, dim=2)

        if show:
            show_timg(grids, title=title)
        if to_save:
            out_path = out_path or Path(f'./content_traversal_{now2str()}.png')
            torchvision.utils.save_image(grids, out_path)
        model.train(is_training)


def run_style_traversal(
        model: pl.LightningModule,
        content_code: torch.Tensor,
        style_code: torch.Tensor,
        traversal_start: Union[float, Iterable[float]],
        traversal_end: Union[float, Iterable[float]],
        n_traversals: int,
        show: bool = True,
        title: str = '',
        to_save: bool = True,
        out_path: Optional[Path] = None,
        verbose: bool = False,
) -> torch.Tensor:
    """Given a fixed content code, traverse the style code each dimension  $j$
    independently, from `traversal_start[j]` to `traversal_end[j]` at
    `n_traversals` steps.

    Parameters
    ----------
        model : pl.LightningModule
        content_code : torch.Tensor; shape (dim_content, )
        style_code : torch.Tensor; shape (dim_style,)
        traversal_start : Iterable; length == dim_style
            a vector of floats that indicate the starting point of the traversal
            for each dimsion of the style code
        traversal_end : Iterable; length == dim_style
            a vector of floats that indicate the ending point of the traversal
            for each dimsion of the style code
        n_traversals : int
            how many traversal steps per dimension
        show : bool
            True to show the result in a grid where $j$th col is the style dim $j$,
            and the $i$th row shows the $i$th step in that direction.
        title : str

    Returns
    -------
        torch.Tensor; shape (`n_traversals`, dim_style, *dim_input_x)
            a batch of reconstructions from each dimension's traversals.
            Eg: output[n][j] contains a (C,H,W) image reconstructed by a $n$th
            step at style_code's dim jth direction with the fixed style code.
    """
    is_training = model.training
    model.eval()
    style_dim = style_code.shape[-1]
    try:
        traversal_start[0]
    except TypeError:
        traversal_start = torch.zeros(style_dim).fill_(traversal_start)
    try:
        traversal_end[0]
    except TypeError:
        traversal_end = torch.zeros(style_dim).fill_(traversal_end)
    with torch.no_grad():
        # Traverse for each dim
        grids = []  # k,v = dim_i, batch of recons while traversing in dim_i direction (n_traversals, *dim_x)
        for dim_i in range(style_dim):
            min_dim_i = traversal_start[dim_i]
            max_dim_i = traversal_end[dim_i]
            print(min_dim_i, max_dim_i)
            traversals = get_traversals(style_code, dim_i, min_dim_i, max_dim_i, n_traversals)

            # Pass to the decoder
            dict_z = {
                "c": content_code.repeat((n_traversals, 1)),
                "s": traversals,
            }
            z = model.combine_content_style(dict_z)
            recons = model.decode(z)

            grid = torchvision.utils.make_grid(recons, nrow=1)  # Caveat: nrow is num of colms!
            grids.append(grid)
        grids = torch.cat(grids, dim=2)

        if show:
            show_timg(grids, title=title)
        if to_save:
            out_path = out_path or Path(f'./style_traversal_{now2str()}.png')
            torchvision.utils.save_image(grids, out_path)
        model.train(is_training)
#         return recons















