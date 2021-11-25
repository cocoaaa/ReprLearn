from argparse import ArgumentParser
from typing import Union, Tuple, Optional, Callable, Dict, Any
from pathlib import Path
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms

from reprlearn.data.datasets.mono_mnist import MonoMNIST
from .base_datamodule import BaseDataModule

class MonoMNISTDataModule(BaseDataModule):
    """Subset of MNIST dataset containing a monochrome image tensors
    Use it to make a datamodule for either Gray, Red, Green, Blue mnist (sub) datasets
    """
    def __init__(self, *,
                 data_root: Union[Path, str],
                 color: str,
                 seed: int=123,
                 in_shape: Tuple,
                 batch_size: int,
                 colorstr_transform: Optional[Callable] = None,
                 pin_memory: bool = True,
                 num_workers: int = 16,
                 shuffle: bool = True,
                 verbose: bool = False,
                 **kwargs
                 ):
        """
        :param data_root: Root dir that contains "mnist_{color}.pkl" files
        :param color: One of gray, red, green, blue
        :param seed: Seed that was/will be used to split the original MNIST into 4 subsets.
            Also, used to split the full monochrome dataset into train,val
        :param in_shape:
        :param batch_size:
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
        self.color = color.lower()
        assert self.color in ["gray", "red", "green", "blue"], "color must be one of gray, red, green, blue"

        self.seed = seed
        self.n_classes = 10

        # Set the following to make this class compatible with MultiMonoMNIST
        # self.n_contents = n_classes
        # self.n_styles = 1

        # self.n_train = kwargs.get('n_train', 55000) # set in `setup` method
        # self.n_val = kwargs.get('n_val', 5000) # set in `setup` method
        self.train_mean = torch.tensor([0.1307, ])
        self.train_std = torch.tensor([0.3081, ])

        # Extra trnansforms to do on top of MonoMNIST's base transforms
        self.transform = transforms.Compose([
            transforms.Resize(in_shape[-2:]),
            transforms.Normalize(self.train_mean, self.train_std)
        ])
        self.colorstr_transform = colorstr_transform
        # Update hparams with MNISTM specifics
        self.hparams.update({
            "color": self.color,
            "seed": self.seed,
            "n_classes": self.n_classes,
        })


    @property
    def name(self) -> str:
        return f"MonoMNIST-{self.color}"
        # return self.full_ds.name # Caution: valid only after `fit` method is run.

    def prepare_data(self):
        # If needed, save and split the original MNIST data into 4 subsets
        if not self.data_root.exists():
            # Running with any color(eg. "red") will have the same, desired effect
            MonoMNIST(self.data_root, "red", download=True, seed=self.seed,
                      train=True)
            MonoMNIST(self.data_root, "red", download=True, seed=self.seed,
                      train=False)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.full_ds = MonoMNIST(self.data_root,
                                color=self.color,
                                seed=self.seed,
                                train=True,
                                transform=self.transform,
                                 colorstr_transform=self.colorstr_transform,
                                     )
            self.n_train = int(len(self.full_ds)*0.7)
            self.n_val = len(self.full_ds) - self.n_train
            self.train_ds, self.val_ds = random_split(self.full_ds, [self.n_train, self.n_val],
                                                      generator=torch.Generator().manual_seed(self.full_ds.seed))

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_ds = MonoMNIST(self.data_root,
                                               color=self.color,
                                               seed=self.seed,
                                               train=False,
                                               transform=self.transform,
                                               colorstr_transform=self.colorstr_transform,
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
    def unpack(batch: Dict[str,Any]) -> Tuple[Any,Any,Any]:
        # Delegate it to its Dataset's classmethod unpack
        return MonoMNIST.unpack(batch)

    @staticmethod
    def add_model_specific_args(parent_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
        # override existing arguments with new ones, if exists
        if parent_parser is not None:
            parents = [parent_parser]
        else:
            parents = []

        parser = ArgumentParser(parents=parents, add_help=False, conflict_handler='resolve')
        parser.add_argument('--data_root', type=str, default='./')
        parser.add_argument('--color', type=str, required=True,
                            help="Color of the background of MNIST. Must be one of red, green, blue, and gray")
        parser.add_argument('--seed', type=int, default=123,
                            help='Set random seed for spliting MNIST into 4 monochrome subsets')
        parser.add_argument('--in_shape', nargs=3, type=int, default=[3, 32, 32])
        parser.add_argument('-bs', '--batch_size', type=int, default=32)
        parser.add_argument('--pin_memory', action="store_true", default=True)
        parser.add_argument('--num_workers', type=int, default=16)
        parser.add_argument('--shuffle', type=bool, default=True)

        return parser
