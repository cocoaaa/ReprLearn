from typing import Union, Optional
from pillow import Image
from pathlib import Path


# load image as rgb
def load_rgb(img_fp):
    img = Image.open(img_fp).convert('RGB') # use 'L' for grayscale
    return img

   
# src: torchvision.datasets.folder.py
def pil_loader(path: Union[Path,str]) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
