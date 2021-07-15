from argparse import ArgumentParser
from typing import Union, Tuple, Optional, Callable
from pathlib import Path
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms

from src.data.datasets.rotated_mnist import RotatedMNIST
from .base_datamodule import BaseDataModule

class RotatedMNISTDataModule(BaseDataModule):
    """A datamodule for a (homogeneous, ie. one rotation angle) Rotated MNIST dataset
    """

    def __init__(self, *,
                 data_root: Union[Path, str],
                 angle: float,
                 in_shape: Tuple,
                 batch_size: int,
                 angle_label_transform: Optional[Callable] = None,
                 split_seed: Optional[int] = None,
                 pin_memory: bool = True,
                 num_workers: int = 16,
                 shuffle: bool = True,
                 verbose: bool = False,
                 **kwargs
                 ):
        """

        :param data_root: Root dir that contains "mnist_{color}.pkl" files
        :param angle: One of gray, red, green, blue
        :param seed: Seed that was/will be used to split the original MNIST into 4 subsets.
            Also, used to split the full monochrome dataset into train,val
        :param in_shape:
        :param batch_size:
        :param angle_label_transform:
        :param split_seed: seed used to split train/val data
        :param pin_memory:
        :param num_workers:
        :param shuffle:
        :param verbose:
        :param kwargs:
        """

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
        self.angle = angle

        # Set the following to make this class compatible with MultiMonoMNIST
        # self.n_contents = n_classes
        # self.n_styles = 1

        # self.n_train = kwargs.get('n_train', 55000) # set in `setup` method
        # self.n_val = kwargs.get('n_val', 5000) # set in `setup` method
        self.train_mean = torch.tensor([0.1307, ])
        self.train_std = torch.tensor([0.3081, ])

        # Extra transforms to do on top of RotatedMNIST's base transforms (ie. ToTensor & TF.rotate)
        self.transform = transforms.Compose([
            transforms.Resize(in_shape[-2:]),
            transforms.Normalize(self.train_mean, self.train_std)
        ])
        self.digit_label_transform = None
        self.angle_label_transform = kwargs.get(angle_label_transform, None)
        self.split_seed = split_seed
        # Update hparams with this DataModule specifics
        self.hparams.update({
            "angle": self.angle,
            "n_classes": self.n_classes,
        })


    @property
    def name(self) -> str:
        return self.full_ds.name

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.full_ds = RotatedMNIST(
                self.data_root,
                angle=self.angle,
                transform=self.transform,
                digit_label_transform=self.digit_label_transform,
                angle_label_transform=self.angle_label_transform,
                train=True,
                download=True
            )
            self.n_train = int(len(self.full_ds)*0.7)
            self.n_val = len(self.full_ds) - self.n_train
            self.train_ds, self.val_ds = random_split(self.full_ds, [self.n_train, self.n_val],
                                                      generator=torch.Generator().manual_seed(self.split_seed))

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_ds = RotatedMNIST(
                self.data_root,
                angle=self.angle,
                transform=self.transform,
                digit_label_transform=self.digit_label_transform,
                angle_label_transform=self.angle_label_transform,
                train=False,
                download=True
            )

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
