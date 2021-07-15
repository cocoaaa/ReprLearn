"""
Base Dataset class for datasets with two kinds of labels, "content label" and "style label".
The content label is compatiable with the target label in a standard dataset for a supervised learning problem.
The style label is compatible with the domain label in a standard dataset for a domain-adaptation problem.

A two-factor dataset returns an item as a dictionary with keys:
- "x" (required): torch.Tensor
- "<name_of_content_class>": eg. "digit" for Colorized MNIST dataset
- "<name_of_style_class>" : eg. "color" for Colorized MNIST dataset

Any subclass must implement:
- self.__getitem__(idx)
- self.unpack(batch) -> Tuple of the values of a sample from this Dataset object
    where each sample is a dictionary, returned from the __getitem__ method.
    Assumes that the inheriting class has its class property, _keys, is set to
    the keys of this dictionary.

    E.g. if a sample is {"x": timg, "y": "8321_023", "d": "osm"},
    and in conjunction with the torch DataLoader's collatefn (which creates a batch of samples,
    while maintaining the dictionary structure so that batch is a {"x": timg_batch, "y": list-of-str, "d": list of string}.

    self.unpack(batch) returns:
        - x is the main data input (eg. an image),
    y is the content-label, and
    d is the domain/style label
- self.keys()
Examples
--------


"""
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Set, Any, Dict, Tuple
from torch.utils.data import Dataset
import pytorch_lightning as pl

class TwoFactorDataset(Dataset):

    _keys = None # Required : List[str]

    def __init__(self):
        super().__init__()
    #todo: consider adding self.n_contents, self.n_styles

    def __getitem__(self, item:int) -> Dict[str, Any]:
        """Must return a dict that has required key,value pairs:
        - "x": (torch.Tensor) of a single datapoint; Required
        - "<name_of_content_class>": eg. "digit"
        - "<name_of_style_class>": eg. "color" for MonoMNIST dataset or, "style" for Maptiles dataset
        """
        raise NotImplementedError

    def get_x_shape(self):
        _temp = self[0]
        _x, _y, _d = self.unpack(_temp)
        return _x.shape

    def get_content_style_reps(self: "TwoFactorDataset",
                               n_contents: int,
                               n_styles: int,
    ) -> np.ndarray:
        """Returns a 2d matrix whose rows are content-labels, and cols are style-labels
        e.g. out[content_id, style_id] is an tensor image with digit0, style(angle)=0

        todo: make a class of "reps" of each dataset (e.g. MNISTr)
        - implement this function as a method
        - add a function that grabs the image of content_label i and style_label j
        """
        in_shape = self.get_x_shape()
        h, w = in_shape[-2:]

        reps = np.zeros((n_styles * h, n_contents * w))
        is_collected = np.zeros((n_styles, n_contents))
        for i in range(len(self)):
            if is_collected.all():
                break

            sample = self[i]
            x, y, d = self.unpack(sample)
            #         if isinstance(label_c, torch.Tensor):
            #             label_c = label_c.item()
            reps[h*d : h*(d+1), w*y : w*(y+1)] = x.numpy()
            is_collected[d, y] = True

        return reps

    def select_a_content_style_repr(
            self: "TwoFactorDataset",
            reps: np.ndarray,
            d: int,  # style-label
            y: int,  # content-label
    ) -> np.ndarray:
        """Given the 2-dim (if each image is grayscale; if color 3-dim) np.array that is
        a collection of representative/random sample image of each contnet (row) and each style (col),
        return the image as np.array of the sample image of content_label=y and style_label=d
        """
        # todo: check if d and y are in valid range

        in_shape = self.get_x_shape()
        h, w = in_shape[-2:]
        return reps[h*d : h*(d+1), w*y : w*(y+1)]

    @classmethod
    def keys(cls) -> List[str]:
        """Returns a list of keys of an item (which is a dictionary) in the dataset
        """
        # raise NotImplementedError
        return cls._keys

    @classmethod
    def unpack(cls, batch: Dict[str, Any]) -> Tuple[Any]:
        if cls.keys() is None:
            raise NotImplementedError

        return [batch[k] for k in cls.keys()]


