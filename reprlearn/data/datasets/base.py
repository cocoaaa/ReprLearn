import os
from collections import defaultdict
from typing import Iterable, Optional, Callable, List, Dict, Union, Tuple, cast
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import has_file_allowed_extension, pil_loader
from torchvision.datasets import DatasetFolder, ImageFolder
import joblib
from reprlearn.utils.misc import has_img_suffix, is_img_fp, is_valid_dir
from reprlearn.data.datasets.utils import create_subdf
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


class ImageDatasetFromDF(Dataset):
    """A dataset that loads (img, label) from a given pd.DataFrame of a structure:
    | (index) | img_fp_col | label_col | ...
    |automatic| /dataset/dog/img0001.png | dog | ...
    
    Args:
    - img_fp_colname (str): name of column containing each datapoint's filepath
    - class_colname (str): name of column specifying the class/label names
    - max_n_per_class (int, default None): number of images to load from each class
        If None, use all images in each class.
    
    Note: 
    - we sort each class's image filepaths using sorted(),
    and, if shuffle, apply np.random.shuffle to each row 
    and, the take the first max_n_per_class to create the df.
    """
    def __init__(self, df: pd.DataFrame,
                 img_fp_colname: str,
                 class_colname: str,
                 shuffle: bool,
                 seed:Optional[int]=123,
                 max_n_per_class: Optional[int]=None,
                 xform:Optional[Callable]=None,
                 target_xform:Optional[Callable]=None,
                ):
        cols = df.columns
        assert img_fp_colname in cols, "img_fp_colname must be in df.columns"
        assert class_colname in cols, "class_colname must be in df.columns"
        self.df = df 
        self.class_colname = class_colname
        self.img_fp_colname = img_fp_colname
        self.max_n_per_class = max_n_per_class # if None, use all images in each class
        self.xform = xform
        self.target_xform = target_xform
        
        # Dataset attributes
        self.classes = sorted(self.df[class_colname].unique())
        self.class_to_idx = {k:i for i,k in enumerate(self.classes)} #class name: str->index
        self.c2i = self.class_to_idx
        self.i2c = {i:k for i,k in enumerate(self.classes)} # class name: idx->str
        
        
        # make a subset of the df_all by taking max_n_per_class rows from each class group
        self.df = create_subdf(self.df, 
                               col_groupby=class_colname, 
                               col_sortvalues=img_fp_colname,
                               max_n_per_group=max_n_per_class,
                               shuffle=shuffle,
                               seed=seed 
                               )
        
        
    def __len__(self) -> int:
        return len(self.df)
    
    
    def __getitem__(self, index) -> Tuple:
        row = self.df.iloc[index]
        img_fp = row[self.img_fp_colname]
        class_name = row[self.class_colname]
        class_idx = self.c2i[class_name]
        
        # img = Image.open(img_fp)
        img = pil_loader(str(img_fp)) #loads pil image in rgb. handles rgba. 
        if self.xform is not None:
            img = self.xform(img)
        if self.target_xform is not None:
            class_idx = self.target_xform(class_idx)
        return (img, class_idx)
    
        
        
        
            
        

class FlatImageSourceDataset(Dataset):
    """ Create dataset of images given the image folder 
    Assumes data_dir has the structure of:
        <data_dir>
        | - xxx.png
        | - xxy.png
        | -...
    Each datapoint is of a tuple of (image, img_filename)

    Args:
    ----
    img_dir : (Path) path to the image folder
    transform : (Optional[callable]) applied to the image if not None

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
        # return img #, self.img_fps[idx].name #2022-07-24
        return img, self.img_fps[idx].name 
 


    def collect_all_timgs(self,
                      batch_size:int=64) -> torch.Tensor:
        """Collect all images in the folder into a single tensor (n_imgs, nc, h, w).
        
        Returns:
        -------
        all_timgs : (Tensor) of shape (n_imgs, nc, h, w)
        """
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
        """
        Args
        ----
        out_fp : (Path) filepath to the output pickled file of a torch.Tensor (n_imgs, nc, h, w)
        """
        all_timgs = self.collect_timgs(batch_size)
        joblib.dump(all_timgs, out_fp)
        print("Saved pickled file of all timgs: ", out_fp)
        return out_fp



class MulticlassDataset10k(ImageFolder):
    max_n_data_per_class = 10_000
    shuffle_filenames = True
    random_seed = 0
    
    @staticmethod
    def make_dataset(
        directory,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = MulticlassDataset10k.find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):

                if MulticlassDataset10k.shuffle_filenames:
                    np.random.seed(MulticlassDataset10k.random_seed)
                    np.random.shuffle(fnames)

                n_added_per_class = 0
                for fname in sorted(fnames):
                    if n_added_per_class >= MulticlassDataset10k.max_n_data_per_class: #todo: consider when max_n_data_per_class is a dict
                        break
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
                        n_added_per_class += 1

                        if target_class not in available_classes: # available_class as long as there is one datapoint of the class
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances



class MulticlassDataset10k(ImageFolder):
    max_n_data_per_class = 10_000
    shuffle_filenames = True
    random_seed = 0
    
    @staticmethod
    def make_dataset(
        directory,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = MulticlassDataset10k.find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):

                if MulticlassDataset10k.shuffle_filenames:
                    np.random.seed(MulticlassDataset10k.random_seed)
                    np.random.shuffle(fnames)

                n_added_per_class = 0
                for fname in sorted(fnames):
                    if n_added_per_class >= MulticlassDataset10k.max_n_data_per_class: #todo: consider when max_n_data_per_class is a dict
                        break
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
                        n_added_per_class += 1

                        if target_class not in available_classes: # available_class as long as there is one datapoint of the class
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances

class DatasetFromDF(Dataset):
    """Dataset class that contains images and labels read off from the data 
    in the given pd.DataFrame.
    e.g.: df is of form:
    
    |   img_fp           | fam_name | model_name |
    | /vae-nvae/0000.png | vae      | nvae       |
    | ...
    |/gan-ddgan/0000.png | gan      | ddgan      |
    | ...
    
    
    Args:
    - df_all (pd.DataFrame) : dataframe containing image filepaths and labels
    - label_key (str)       : column name of df_all that will be used as label of each image
    - xform (Callable)      : transformation applied to datapoint (image)
    - target_xform (Callable): transformation applied to label index
    """
    def __init__(self,
                 df: pd.DataFrame,
                 label_key: str,
                 xform: Optional[Callable]=None,
                 target_xform: Optional[Callable]=None
                ):
        self.df = df
        self.label_key = label_key
        assert self.label_key in self.df.columns
        
        self.label_set = sorted(df[label_key].unique())
        self.c2i = {c:i for i,c in enumerate(self.label_set)}
        self.i2c = {i:c for c,i in self.c2i.items()}
        self.xform = xform
        self.target_xform = target_xform
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int)-> Tuple:
        """ returns xform(img), target_xform(label)
        """
        img_fp = self.df.iloc[idx]['img_fp']
        img = Image.open(img_fp).convert('RGB')
        
        if self.xform is not None:
            img = self.xform(img)

        label_str = self.df.iloc[idx][self.label_key]
        label = self.c2i[label_str]
        if self.target_xform is not None:
            label = self.target_xform(label)
        
        return img, label

    
    