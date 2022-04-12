from collections import defaultdict
from typing import Iterable, Optional, Callable, List, Dict, Union
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
from reprlearn.utils.misc import has_img_suffix
from sklearn.model_selection import StratifiedShuffleSplit

class ImageDataset(Dataset):

    def __init__(self,
                 imgs: np.ndarray,
                 targets: Union[torch.IntTensor, List[int]],
                 img_xform: Optional[Callable] = None):
        """
        Args
        ----
        imgs : np.ndarray
            Array of uint8 images; (N, 32, 32, 3)
        targets : List[int]
            List of integers indicating classes
        img_xform : Optional[Callable]
            Transformation to be applied to each item. First `transform` object
            applied should work with input of type `np.ndarray` (h,w,nc)
        """
        super().__init__()
        self.imgs = imgs
        self.targets = targets
        self.unique_classes = np.unique(self.targets)
        self.img_xform = img_xform

    def __getitem__(self, ind: int):
        img = self.imgs[ind]
        target = self.targets[ind]
        if self.img_xform is not None:
            img = self.img_xform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

# helper to create an ImageDataset object given classes indices
def create_dset_of_classes(imgs: np.ndarray,
                           classes: Iterable[int],
                           selected_classes: List[int],
                           img_xform: Optional[Callable] = None) -> Dataset:
    """Given an array of imgs, create a dataset that is of one of the given classes
    Args
    ----
    imgs : np.ndarray
    classes : List[int]
    selected_classes : iterable[int]
        List of class indices to constructe the dataset; i.e, the returned Dataset
        contains all images of these classes, and no images of any other classes

    Returns
    -------
        (ImageDataset) containing a subset of `imgs` that are of one of the classes
        in `classes`
    """
    is_member = [c in selected_classes for c in classes]
    return ImageDataset(imgs=imgs[is_member],
                        targets=classes[is_member],
                        img_xform=img_xform)


class ImageFolderDataset(Dataset):
    """ returns  (image, img_filename)
    transform: optional[callable]: applied to the image if not None

    """
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_fps = [
            img_fp for img_fp in img_dir.iterdir()
            if img_fp.is_file() and has_img_suffix(img_fp)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.img_fps)

    def __getitem__(self, idx):
        img = Image.open(self.img_fps[idx])

        if self.transform is not None:
            img = self.transform(img)
        return img, self.img_fps[idx].name


    def collect_all_timgs(self,
                      batch_size:int=64) -> torch.Tensor:
        dl = DataLoader(self, batch_size=batch_size)
        all_timgs = []
        for batch in dl:
            batch_imgs, batch_fns = batch
            all_timgs.append(batch_imgs)
        all_timgs = torch.stack(all_timgs, dim=0)
        return all_timgs


    def pickle_all_imgs(self,
                        batch_size: int,
                        out_fp: Path,
    ) -> Path:
        """Assumes data_dir has the structure of:
        <data_dir>
        | - xxx.png
        | - xxy.png
        | -...
        Returns
        -------
            path to the output pickled file of a torch.Tensor (n_imgs, nc, h, w)
        """
        all_timgs = self.collect_timgs(batch_size)
        joblib.dump(all_timgs, out_fp)
        print("Saved pickled file of all timgs: ", out_fp)
        return out_fp







