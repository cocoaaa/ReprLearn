import os
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional
from xml.dom.pulldom import default_bufsize
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from reprlearn.data.datasets.base import SingleImageSourceDataset
"""
2022-07-24: wip 
- Look at datasets.base.ImageFolderDataset for dataset class for training gms on celeba
"""

def get_my_celeba64_dataset(
    img_dir: Optional[Path]=None,
    center_crop_size: int=140,
    resize: int=64):
    """Get a dataset for CelebA64 with the transform of with the given center_crop size
    and the target image size after the transforms applied.
    img_dir is set to either:
        - Path to the image dir of the original celebA (`img_aligned`) folder
        if `is_preprocessed` is True.
        - or, the image dir which contains iamges already preprocessed with 
        the center-crop 140 and resize to 64.

    Args:
    -----
    - center_crop_size: int
        Size of the center cropped image
    - resize: int
        size of the iamge after the whole xform of (center-crop and resizing)

    Note:
    -----
    As of 2022-07-24 I'm using it to train the following models:
    - prog-gan
    
    I think I have also used the same preprocessing for training the GAN suites,
    implemented by LynnHo.
    Also probably the beta-vae, and dfc-vae? #to-verify
    """

    img_dir = img_dir or Path('/data/datasets/reverse-eng-data/originals/CelebA/img_align_celeba')
    xform = transforms.Compose([
    transforms.CenterCrop(center_crop_size),
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    ])

    return SingleImageSourceDataset(img_dir=img_dir, transform=xform)


def get_my_celeba64_dataloader(
    *,
    img_dir: Optional[Path]=None,
    center_crop_size: int=140,
    resize: int=64,
    **kwargs,
):
    dl_kwargs = {
        'shuffle': True,
        'drop_last': True,
        'pin_memory': True
    }
    dl_kwargs.update(kwargs)
    dset = get_my_celeba64_dataset(img_dir, center_crop_size, resize)
    return DataLoader(dset, **dl_kwargs)
