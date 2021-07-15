from argparse import ArgumentParser
from typing import Union, Tuple
from pathlib import Path
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from src.data.datasets.mnistm import MNISTM
from .base_datamodule import BaseDataModule

class MNISTMDataModule(BaseDataModule):

    def __init__(self, *,
                 data_root: Union[Path, str],
                 in_shape: Tuple,
                 batch_size: int,
                 pin_memory: bool = True,
                 num_workers: int = 16,
                 shuffle: bool = True,
                 verbose: bool = False,
                 **kwargs
                 ):
        super().__init__(
            data_root=data_root,
            in_shape=in_shape,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            verbose=verbose,
            **kwargs
        )

        # Set attributes specific to this dataset
        self.n_classes = 10
        self.n_train = kwargs.get('n_train', 55000)
        self.n_val = kwargs.get('n_val', 5000)
        self.train_mean = torch.tensor([0.4639, 0.4676, 0.4199])
        self.train_std = torch.tensor([0.2534, 0.2380, 0.2618])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(in_shape[-2:]),
            transforms.Normalize(self.train_mean, self.train_std)
        ])

        # Update hparams with MNISTM specifics
        self.hparams.update({
            "n_classes":  self.n_classes,
                             })


    @property
    def name(self) -> str:
        return 'MNIST-M'

    def prepare_data(self):
        # download
        MNISTM(self.data_root, train=True, download=True)
        MNISTM(self.data_root, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            full_ds = MNISTM(self.data_root, train=True, transform=self.transform)
            self.train_ds, self.val_ds = random_split(full_ds, [self.n_train, self.n_val])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_ds = MNISTM(self.data_root, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle,
                          pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          pin_memory=self.pin_memory, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          pin_memory=self.pin_memory, num_workers=self.num_workers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_root', type=str, default='./')
        parser.add_argument('--in_shape', nargs=3, type=int, default=[1, 32, 32])
        parser.add_argument('-bs', '--batch_size', type=int, default=32)
        parser.add_argument('--pin_memory', action="store_true", default=True)
        parser.add_argument('--num_workers', type=int, default=16)
        parser.add_argument('--shuffle', type=bool, default=True)

        return parser
