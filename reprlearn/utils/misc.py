import inspect
from datetime import datetime
import math
from collections import defaultdict
from functools import partial
from re import I
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy as sp
from skimage.color import rgb2gray
from skimage.transform import resize

from pprint import pprint
import torch
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision.transforms import ToTensor
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable
import warnings


def now2str(delimiter: Optional[str]='-'):
    now = datetime.now()
    now_str = now.strftime(f"%Y%m%d{delimiter}%H%M%S")
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



# image read io
def read_image_as_tensor(img_fp: Path, as_gray:Optional[bool]=False) -> torch.Tensor:
    # todo: test this function
    if as_gray:
        return ToTensor()(Image.open(img_fp).convert('L')) 
    else:
        return ToTensor()(Image.open(img_fp))
        


# ===
# === tensor, npimg, pil_img conversions
# ===
# npimg <--> torch image conversion
# https://www.programmersought.com/article/58724642452/

def npimg2timg(npimg: np.ndarray) -> torch.Tensor:
    if npimg.dtype == np.uint8:
        npimg = npimg / 255.0

    return torch.from_numpy(npimg.transpose((2,0,1)))


def timg2npimg(timg: torch.Tensor) -> np.ndarray:
    return timg.detach().numpy().squeeze().transpose((1,2,0))

def timg2pil(timg: torch.Tensor) -> Image:
    return tv.transforms.ToPILImage()(timg)


def npimgs2timgs(npimgs: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((npimgs.transpose((0,-1, 1, 2))))


def timgs2npimg2(timgs: torch.Tensor) -> np.ndarray:
    """ Converts a batch of tensor images (size is (bs, nC, h, w))
    to an nparray of npimgs (size is (bs, h, w, nC).
    
    Note: If the tensor is in gpu, it will be detached and 
    then converted to numpy.
    Note: we return the numpy image array in the same device as the input tensor.
    
    
    """
    return timgs.detach().numpy().transpose(0, -2, -1, -3)


# ===
# === Array indexing conversion
# ===
# Helper: i -> (r_idx, c_idx) for given span

def get_qr(x:int, divider:int) -> Tuple[int, int]:
    """ Get quotient and remainder when dividing x by divier
    i = span * r_idx + c_idx
    
    c_idx = i % span
    r_idx = (i - c_idx) / span
    
    Usage: 
    -------
    Convert index in 1-dim array to 2-dim array of size (nrows, ncols=span), i.e.,
    i -> (r_idx, c_idx) for given span
    
    Similar: 
    --------
    np.divmod return (x // mod, x % mod)
    """
    r = x % divider
    q = ( x - r ) / divider
    
    return int(q), int(r)

# ===
# === ops on path 
# ===
def mkdir(p: Path, parents=True):
    if not p.exists():
        p.mkdir(parents=parents)
        print("Created: ", p)

       
        
def search_and_move_file(src_dir: Path, search_word: str, out_dir: Path) -> int:
    """ Search any file whose name contains `search_word` in the `src_dir`,
    and move it to `out_dir`
    
    Returns
    -------
    int : number of files moved
    """
    n_moved = 0
    for img_fp in src_dir.iterdir():
        if search_word in str(img_fp):
            fn = img_fp.name # <filename>.<ext>, e.g.: `mydog001.jpg`
            
            # move this file to out_dir folder
            new_fp = img_fp.rename(out_dir/fn)
            print(f'Renamed: {img_fp.absolute()} --> {new_fp.absolute()}')
            n_moved += 1
    return n_moved


# === ops on image dir
def has_img_suffix(fp: Path, valid_suffixes:List[str]=['.png', '.jpeg', '.jpg']):
    return fp.suffix.lower() in valid_suffixes

def is_img_fp(fp:Path, valid_suffixes:List[str]=['.png', '.jpeg', '.jpg']):
    return fp.is_file() and has_img_suffix(fp, valid_suffixes)

def is_valid_dir(fp: Union[Path,str]) -> bool:
    if isinstance(fp, Path):
        fp = str(fp) 
    return not (fp.startswith('.') and Path(fp).is_file())


def count_imgs(dir_path: Path, valid_suffixes:List[str]=['.png', '.jpeg', '.jpg']) -> int:
    """ Count the number of images in the directory """
    c = 0
    for img_fp in dir_path.iterdir():
        if img_fp.is_file() and has_img_suffix(img_fp, valid_suffixes):
            c += 1
    return c


def count_files(dir_path: Path, 
                is_valid_fp: Optional[Callable[[Path],bool]]=None) -> int:
    """ Count the number of files in the directory that satisfies 
    the given validity condition 
    
    """
    c = 0
    for fp in dir_path.iterdir():
        if not fp.is_file():
            continue
        if is_valid_fp is not None and not is_valid_fp(fp):
            continue
        c += 1
    return c

def count_files_in_subdirs(
    root_dir: Path,
    is_valid_fp: Optional[Callable[[Path],bool]]=None,
    verbose: Optional[bool]=True) -> Dict[str,int]:
    """
    Count files in each subdir that satisfies the validity condition.
    
    Returns:
    -------
    counts (Dict[subdir_name, count_of_valid files])
    
    """
    counts = dict()
    for sub_dir in root_dir.iterdir():
        if not is_valid_dir(sub_dir):
            continue
        dirname = sub_dir.name
        n_files =  count_files(sub_dir, is_valid_fp=is_valid_fp)
        counts[dirname] = n_files
        
        if verbose:
            print(f"{dirname}:  {n_files} num. of npy files")
    if verbose:
        print('Done!')
    return counts
    

count_imgs_in_subdirs = partial(
    count_files_in_subdirs, is_valid_fp=is_img_fp
)

count_npys_in_subdirs = partial(
    count_files_in_subdirs, is_valid_fp=lambda fp:  fp.suffix == '.npy' 
)
    

def info_img_dir(img_dir: Path) -> Dict:
    """Get basic stat of the img_dir:
    
    Returns a dict. of (k,v):
    - "n_imgs": Int[number of images]
    - "img_sizes": Dict[ImageSize, Int[count of imgs of that size]]
        - ImageSize=Tuple[int]
    """
    d_stat = dict() # global container
    d_sizes = defaultdict(int)  # contains nimgs per each size
    n_imgs = 0 # total count of images
    for img_fp in img_dir.iterdir():
        if not (img_fp.is_file() and has_img_suffix(img_fp)):
            continue
        npimg = np.array(Image.open(img_fp))
        img_size = npimg.shape #tuple
        d_sizes[img_size] += 1

        n_imgs += 1

    d_stat['n_imgs'] = n_imgs
    d_stat['img_sizes'] = d_sizes

    return d_stat

def get_first_img_info(img_dir: Path) -> np.ndarray:
    
    for img_fp in img_dir.iterdir():
        if img_fp.is_file():
            suffix = img_fp.suffix
            npimg = np.array(Image.open(img_fp))
            print(f'{img_dir.stem}: {npimg.shape}, {suffix}')
            return npimg
        
def get_ith_img_fp(img_dir: Path, ind: int) -> Path:
    """Return filepath to the ith image in the ascending sorted order in `img_dir`
    Note :
    - `ind` is zero-based index.
    - If `ind` is larger than the last index in `img_dir`, filepath to the last 
    image is returned.

    
    """
    i = 0
    img_fps = sorted(
        [img_fp for img_fp in img_dir.iterdir()
        if img_fp.is_file() and has_img_suffix(img_fp)]
        )
    if ind >= len(img_fps):
        ind = - 1 # if ind is out of bound, return the last element in the list of img paths
    return img_fps[ind]

    
                    
def get_ith_npimg(img_dir: Path, ind: int, show:bool=True) -> np.ndarray:
    img_fp = get_ith_img_fp(img_dir, ind)
    pil_img = Image.open(img_fp)
    npimg = np.array(pil_img)
    if show:
        plt.title(ind)
        plt.imshow(npimg)
        plt.axis('off')
    return npimg

get_first_img_fp = partial(get_ith_img_fp, ind=0)
get_first_npimg = partial(get_ith_npimg, ind=0)
get_last_img_fp = partial(get_ith_img_fp, ind=-1)
get_last_npimg = partial(get_ith_npimg, ind=-1)

def get_last_modified_file(dirpath: Path) -> Tuple[Path,str]:
    """Get the path to the file last modified in the dirpath, with the timestamp
    Src: https://realpython.com/python-pathlib/
    """    
    from datetime import datetime
    fp, ts = max((p.stat().st_mtime, p) for p in dirpath.iteridr())
    return (fp, datetime.fromtimestamp(ts))


def make_thumbnail(sample_dir: Path, 
                   out_dir: Path=None, 
                   fn_suffix: str='',
                   n_show: int=64, 
                   n_rows: int=8,
                  ) -> None:
    """Randomly select `n_show` number of images from `sample_dir` and create a
    collage (.png).
    Used to generate thumbnail for synthetic images of our GM dataset, created for 
    NeurIPS22.
    
    
    Naming pattern for generated thumbnail png:
    ===============
    thumb-{model_name}-fn_suffix.png
    
    Args:
    ===============
    - fn_suffix: suffix of the filename 
        If not specified, use current timestamp
    
    Assumption:
    ================
    - `sample_dir`'s parent folder indicates the name of the model so that:
        - <model_dir>
            |- <sample_dir_stem>
            |- thumb: this will be created if not existing. This is the folder to save the thumbnail image.
                |- thumb-{model_name}-v{version}.png : this is the image file of thumbnail this function creates. 
    -  there are more than 128k images #todo: fix it
    
    """
    def to_tensor(pil_img)->torch.Tensor:
        return tv.transforms.ToTensor()(pil_img)   

    model_name = sample_dir.parent.name
    if out_dir is None:
        out_dir = sample_dir.parent/'thumbs'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    
    timgs = []
    for i, img_fp in enumerate(sample_dir.iterdir()):
        if i == n_show: 
            break
        img = Image.open(img_fp)
        timg = to_tensor(img)
        timgs.append(timg)
    # make into one figure/collage
    
    
    # save the thumbnail/collage
    fn_suffix = fn_suffix or now2str()
    out_fp = f'{out_dir}/thumb-{model_name}-{fn_suffix}.png'
    tv.utils.save_image(timgs, out_fp)
    print("Saved thumbnail to: ", out_fp)
    