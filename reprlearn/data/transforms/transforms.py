# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import math
import random
import torch
from torchvision import transforms as T
import numpy as np
from sklearn.preprocessing import minmax_scale
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable
from .functional import  unnormalize, to_monochrome

def to_3dim(X: torch.Tensor, target_size: Tuple[int, int, int], dtype=torch.float32) -> torch.Tensor:
    """
    Rearragne data matrix X of size (n_styles*dim_x, n_contents)
    to (n_styles, n_contents, dim_x)

    Args:
    - X: torch.Tensor of 2dim data matrix
    - target_size: tuple of n_style, n_contents, dim_x
    """
    assert X.ndim == 2
    n_styles, n_contents, dim_x = target_size
    assert X.shape[0] == n_styles * dim_x
    assert X.shape[1] == n_contents

    target = torch.zeros(target_size, dtype=X.dtype)

    for s in range(n_styles):
        for c in range(n_contents):
            img = X[s * dim_x: (s + 1) * dim_x, c]
            target[s, c] = img
    return target.to(dtype)


class Identity: # used for skipping transforms
    def __call__(self, x):
        return x

class Unnormalizer:
    """
    Inverse operation of the channelwise normalization via transforms.Normalizer
    See https://github.com/pytorch/vision/issues/528
    """
    def __init__(self,
                 used_mean: Union[torch.Tensor, np.ndarray],
                 used_std: Union[torch.Tensor, np.ndarray]
                 ):
        self.used_mean = used_mean
        self.used_std = used_std

    def __call__(self, x: Union[torch.Tensor, np.ndarray]):

        return unnormalize(self.used_mean, self.used_std)(x)




class LinearRescaler:
    """
    CAUTION: Breaks the computational graph by returning a new numpy array
    Normalize the input tensor by scaling its values via a linear mapping
    f: [x.min(), x.max()] -> target_range (default: [0,1])
    """

    def __init__(self, target_range:Optional[Tuple[float,float]]=None):
        self.target_range = target_range or [0., 1.]

    def __call__(self, x: Union[np.ndarray, torch.Tensor]):
        shape = x.shape

        out = minmax_scale(x.flatten(), feature_range=self.target_range).reshape(shape)
        if isinstance(x, torch.Tensor):
            out = torch.tensor(out, dtype=x.dtype, device=x.device)
        return out


class Monochromizer:
    """
    Converts a single channel image tensor to a 3-channel mono-chrome image tensor
    - Supports red, green,blue and gray


    """
    def __init__(self, color:str):
        self.color = color.lower()
        assert self.color in ["red", "green", "blue", "gray"]

    @property
    def color2dim(self) ->Dict[str, int]:
        return {"red": 0, "green": 1, "blue": 2}

    def __call__(self, x):
        return to_monochrome(x, self.color)


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img
