import inspect
from datetime import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from skimage.color import rgb2gray
from skimage.transform import resize

from pprint import pprint
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable
import warnings


def now2str():
    now = datetime.now()
    now_str = now.strftime("%Y%m%d-%H%M%S")
    return now_str


def today2str(delimiter: str = "-"):
    # Print today's date (year, month, day) as a string, separated by the
    # `delimiter`
    now = datetime.now()
    return now.strftime(f"%Y{delimiter}%m{delimiter}%d")


def print_mro(x, print_fn:Callable=print):
    """
    Get the MRO of either a class x or an instance x
    """
    if inspect.isclass(x):
        [print_fn(kls) for kls in x.mro()[::-1]]
    else:
        [print_fn(kls) for kls in x.__class__.mro()[::-1]]


def info(arr, header=None):
    if header is None:
        header = "="*30
    print(header)
    print("shape: ", arr.shape)
    print("dtype: ", arr.dtype)
    print("min, max: ", min(np.ravel(arr)), max(np.ravel(arr)))


def mkdir(p: Path, parents=True):
    if not p.exists():
        p.mkdir(parents=parents)
        print("Created: ", p)


def get_next_version(save_dir:Union[Path,str], name:str):
    """Get the version index for a file to save named in pattern of
    f'{save_dir}/{name}/version_{current_max+1}'

    Get the next version index for a directory called
    save_dir/name/version_[next_version]
    """
    root_dir = Path(save_dir)/name

    if not root_dir.exists():
        warnings.warn("Returning 0 -- Missing logger folder: %s", root_dir)
        return 0

    existing_versions = []
    for p in root_dir.iterdir():
        bn = p.stem
        if p.is_dir() and bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace('/', '')
            existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0

    return max(existing_versions) + 1


def get_next_version_path(save_dir: Union[Path, str], name: str):
    """Get the version index for a file to save named in pattern of
    f'{save_dir}/{name}/version_{current_max+1}'

    Get the next version index for a directory called
    save_dir/name/version_[next_version]
    """
    root_dir = Path(save_dir) / name

    if not root_dir.exists():
        root_dir.mkdir(parents=True, exist_ok=True)
        print("Created: ", root_dir)

    existing_versions = []
    for p in root_dir.iterdir():
        bn = p.stem
        if p.is_dir() and bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace('/', '')
            existing_versions.append(int(dir_ver))

    if len(existing_versions) == 0:
        next_version = 0
    else:
        next_version = max(existing_versions) + 1

    return root_dir / f"version_{next_version}"


def get_ckpt_path(log_dir: Path):
    """Get the path to the ckpt file from the pytorch-lightning's log_dir of the model
    Assume there is a single ckpt file under the .../<model_name>/<version_x>/checkpoints

    Examples
    --------
    log_dir_root = Path("/data/hayley-old/Tenanbaum2000/lightning_logs")
    log_dir = log_dir_root/ "2021-01-12-ray/BiVAE_MNIST-red-green-blue_seed-123/version_1"
    ckpt_path = get_ckpt_path(log_dir)
    # Use the ckpt_path to load the saved model
    ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)  # dict object

    """
    ckpt_dir = log_dir / "checkpoints"
    for p in ckpt_dir.iterdir():
        return p


def n_iter_per_epoch(dl:DataLoader):
    n_iter = len(dl.dataset)/dl.batch_size
    if n_iter == int(n_iter):
        return int(n_iter)
    elif dl.drop_last:
        return math.floor(n_iter)
    else:
        return math.ceil(n_iter)

# npimg <--> torch image conversion
# https://www.programmersought.com/article/58724642452/


def npimg2timg(npimg: np.ndarray):
    if npimg.dtype == np.uint8:
        npimg = npimg / 255.0

    return torch.from_numpy(npimg.transpose((2,0,1)))


def timg2npimg(timg: torch.Tensor):
    return timg.detach().numpy().squeeze().transpose((1,2,0))


def npimgs2timgs(npimgs: np.ndarray):
    return torch.from_numpy((npimgs.transpose((0,-1, 1, 2))))


def timgs2npimg2(timgs: torch.Tensor):
    return timgs.detach().numpy().transpose(0, -2, -1, -3)

