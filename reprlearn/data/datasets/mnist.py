# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from typing import Iterable,Tuple
from torchvision.datasets import MNIST

# Add a method to unpack a sample into image and its label
# Similar to TwoFactorDataset's unpack method
@classmethod
def unpack(cls, batch: Iterable) -> Tuple:
    # todo: consider if this is a good design
    # - if batch is a single sample from MNIST Dataset (rather than a batch from
    # MNISTDataModule's any dataloaders, e.g., ) the output of this method
    # is a single image, and a single label (ie. no batch dimension)
    return batch[0], batch[1]

MNIST.unpack = unpack
