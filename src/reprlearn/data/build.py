# encoding: utf-8
"""
Build MNIST datasets and data loaders
"""

from torch.utils import data

from .datasets.mnist import MNIST
from .transforms import build_transforms


def build_dataset(transforms, is_train=True):
    dataset = MNIST(root='./', train=is_train, transform=transforms, download=True)
    return dataset


def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False

    transforms = build_transforms(cfg, is_train)
    dataset = build_dataset(transforms, is_train)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader
