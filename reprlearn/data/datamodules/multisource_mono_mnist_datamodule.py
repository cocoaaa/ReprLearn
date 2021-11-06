from argparse import ArgumentParser
from typing import Union, Tuple, Optional, List
from pathlib import Path
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms

from reprlearn.data.datasets import MultiMonoMNIST
from .multisource_datamodule import MultiSourceDataModule

class MultiMonoMNISTDataModule(MultiSourceDataModule):
    """Subset of MNIST dataset containing a monochrome image tensors
    Use it to make a datamodule for either Gray, Red, Green, Blue mnist (sub) datasets
    """
    _name_formatspec = "MNIST-{colors_str}_seed-{seed}"

    def __init__(self,
                 data_root: Path,
                 colors: List[str],
                 seed: int,
                 in_shape: Tuple,
                 # Dataloading args
                 batch_size: int,
                 pin_memory: bool = True,
                 num_workers: int = 16,
                 shuffle: bool = True,
                 verbose: bool = False,
                 **kwargs
                 ):


        """
        :param data_root: - data_root: root dir that contains "mnist_{color}.pkl" files (ie. 4 subsets of
        train or test MNIST dataset; one subset for each Monochrome MNIST (red, green, blue, gray)
            Eg. data_root = Path('/data/hayley-old/Tenanbaum2000/data/Mono-MNIST/')

        :param color: One of gray, red, green, blue
        :param seed: Seed that was used to split the original MNIST into 4 subsets
         to generate the source datasets of each monochrome (eg. 123)
        In this class, this given seed will be used to split the full dataset into train, val datasets
        :param in_shape:
        :param batch_size:
        :param pin_memory:
        :param num_workers:
        :param shuffle:
        :param verbose:
        :param kwargs:
        """
        colors = [c.lower() for c in colors]
        n_contents = kwargs.get("n_contents", 10)
        super().__init__(
            data_root=data_root,
            n_contents=n_contents,
            source_names=colors,
            in_shape=in_shape,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            verbose=verbose,
            **kwargs
        )

        # Set attributes specific to this dataset
        self.colors = colors
        for c in self.colors:
            assert c in ["gray", "red", "green", "blue"], "color must be one of gray, red, green, blue"
        self.seed = seed
        self.n_contents = n_contents
        self.train_mean = torch.tensor([0.1307, ])
        self.train_std = torch.tensor([0.3081, ])

        # Extra trnansforms to do on top of MonoMNIST's base transforms,
        # which are [torch.ToTensor(), Monochromizer()]
        self.transform = transforms.Compose([
            transforms.Resize(in_shape[-2:]),
            transforms.Normalize(self.train_mean, self.train_std)
        ])

        # Update hparams with this multisource MonoMNIST specifics
        self.hparams.update({
            "seed": self.seed,
        })


    @property
    def name(self) -> str:
        return self._name_formatspec.format(
            colors_str='-'.join(self.colors),
            seed=self.seed
        )


    def prepare_data(self):
        pass

    def setup(self, stage=None, **kwargs):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.full_ds = MultiMonoMNIST(
                data_root=self.data_root,
                colors=self.colors,
                seed=self.seed,
                transform=self.transform,
                train=True,
            )
            n_train = int(len(self.full_ds) * 0.7)
            n_val = len(self.full_ds) - n_train
            self.n_train = kwargs.get('n_train', n_train)
            self.n_val = kwargs.get('n_val', n_val)
            self.train_ds, self.val_ds = random_split(self.full_ds, [self.n_train, self.n_val],
                                                      generator=torch.Generator().manual_seed(self.seed))

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_ds = MultiMonoMNIST(data_root=self.data_root,
                                          colors=self.colors,
                                          seed=self.seed,
                                          transform=self.transform,
                                          train=False
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
        # Required
        parser.add_argument('--data_root', type=str,
                            default='/data/hayley-old/Tenanbaum2000/data/Mono-MNIST/')
        parser.add_argument('--colors', nargs="+", type=str)
        parser.add_argument('--seed', type=int, default=123)
        # Optional
        parser.add_argument('--in_shape', nargs=3, type=int, default=[3, 32, 32])
        parser.add_argument('--n_contents', type=int, default=10)
        parser.add_argument('-bs', '--batch_size', type=int, default=128)
        parser.add_argument('--pin_memory', action="store_true", default=True)
        parser.add_argument('--num_workers', type=int, default=16)
        parser.add_argument('--shuffle', type=bool, default=True)

        return parser
