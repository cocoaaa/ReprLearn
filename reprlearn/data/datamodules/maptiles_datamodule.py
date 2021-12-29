from pathlib import Path
from argparse import ArgumentParser
from typing import Any, Dict, Tuple, Optional, Iterable, Union, Callable

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from reprlearn.data.datasets.maptiles import MaptilesDataset
from reprlearn.data.datamodules import BaseDataModule

class MaptilesDataModule(BaseDataModule):

    def __init__(self, *,
                 data_root: Path,
                 cities: Iterable[str],
                 styles: Iterable[str],
                 zooms: Iterable[str],
                 in_shape: Union[torch.Size, Tuple[int, int, int]], # target image shape
                 batch_size: int,
                 transform: Callable = None,
                 target_transform: Callable = None,
                 df_fns: pd.DataFrame = None,
                 # --end of MaptilesDataset init args (except in_shape, batch_size)

                 pin_memory: bool=True,
                 num_workers: int = 16,
                 shuffle: bool = True,
                 verbose: bool = False,
                 **kwargs):
        """
        :param cities:
        :param styles:
        :param zooms:
        :param transform:
        :param target_transform:
        :param df_fns:
        :param data_root:
        :param in_h: (C,H,W)
        :param in_channels:
        :param bs:
        :param verbose:
        :param pin_memory:
        :param num_workers:
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
        self.df_fns = df_fns
        self.cities = sorted(cities)
        self.styles = sorted(styles)
        self.n_classes = len(self.styles)
        self.zooms = zooms

        # City string <-> city_idx
        self.city2idx = {c:i for i,c in enumerate(self.cities)}
        self.idx2city = {i:c for i,c in enumerate(self.cities)}

        # Style string <-> style_idx
        self.style2idx = {s: i for i, s in enumerate(self.styles)}
        self.idx2style = {i: s for i, s in enumerate(self.styles)}

        # transforms
        self.transform = transform
        self.target_transform = target_transform
        self.n_channels, self.in_h, self.in_w = in_shape

        # default transforms
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.in_shape[-2:]),
            ])

        # Default transsforms for Maptile dataset's target label_dict
        # MaptilesDataset class's `__getitems__` returns (x, label_dict)
        # -- where label_dict = {
        #    "city": city,
        #    "style": style,
        #    "zoom": zoom,
        #    "coord": coord}
        # Default: returns the style label
        # -- ie. prepare a sample for the style prediction problem
        # if self.target_transform is None:
        #     self.target_transform = transforms.Compose([
        #         transforms.Lambda(
        #             lambda label_dict: self.style2idx[label_dict["style"]]
        #         )
        #     ])

        # Update hparams with maptiles specifics
        self.hparams.update({"cities":  self.cities,
                            "styles": self.styles,
                            "zooms": self.zooms,
                             "n_classes": self.n_classes,
                             })


    @classmethod
    def from_maptiles_dataset(cls, dset: MaptilesDataset, *,
                              in_shape: Tuple[int, int, int],
                              batch_size: int = 32,
                              pin_memory: bool = True,
                              num_workers: int = 8,
                              verbose: bool = False,
                              ):
        kwargs = {
            'data_root': dset.data_root,
            'cities': dset.cities,
            'styles': dset.styles,
            'zooms': dset.zooms,
            'transform': dset.transform,
            'target_transform': dset.target_transform,
            'df_fns': dset.df_fns,
            # --end of MaptilesDataset init args
            'in_shape': in_shape,
            'batch_size': batch_size,
            'verbose': verbose,
            'pin_memory': pin_memory,
            'num_workers': num_workers,
        }
        return cls.from_dict(**kwargs)


    @property
    def name(self):
        try:
            return self.train_ds.name
        except:
            return 'Maptiles'

    def on_fit_start(self, *args, **kwargs):
        print(f"{self.__class__.__name__} is called")

    def prepare_data(self):
        # TODO:
        # download maptile dataset to self.data_dir?
        pass

    def setup(self, stage: str, use_training_stat: bool=True):
        """Main method to setup this datamodule. Sets up MaptileDataset objects as
        its self.train_ds, self.val_ds (and self.test_ds)
        -- this also configures this DataModule to have a specified transforms
        -- that will be applied to each sample in the dataset

        Note: This function is called on every GPU in a node/machine
        """
        # breakpoint()
        dset = MaptilesDataset(
            df_fns=self.df_fns,
            data_root=self.data_root,
            cities=self.cities,
            styles=self.styles,
            zooms=self.zooms,
            transform=self.transform,
            target_transform=self.target_transform,
            verbose=self.verbose)

        # split to train/val or test
        if stage == 'fit':
            # Set train_ds, val_ds
            self.train_ds, self.val_ds = MaptilesDataset.random_split(dset, 0.7)
            self.n_train, self.n_val = len(self.train_ds), len(self.val_ds)
            assert len(self.train_ds) + len(self.val_ds) == len(dset)
            print("n_train, n_val: ", self.n_train, self.n_val)

            # Add the channel-wise normalization transform
            self.train_mean, self.train_std = self.train_ds.channel_mean[:self.n_channels], self.train_ds.channel_std[:self.n_channels]
            print("train channelwise_mean,std: ", self.train_mean, self.train_std)

            # Reset the image-transforms for each dataset based on training data's mean, std
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.in_shape[-2:]),
                transforms.Grayscale() if self.n_channels == 1 else torch.nn.Identity(),
                transforms.Normalize(mean=self.train_mean, std=self.train_std),
                ])
            self.train_ds.transform = train_transform

            if use_training_stat:
                self.val_ds.transform = train_transform

            # Reset each dataset's target_transform to be the same as the self.target_transform
            self.train_ds.target_transform = self.target_transform
            self.val_ds.target_transform = self.target_transform


        if stage == 'test':
            # split the whole dataset into tr:val:test=4:3:3
            self.tv_ds, self.test_ds = MaptilesDataset.random_split(dset, 0.7)
            self.train_ds, self.val_ds = MaptilesDataset.random_split(self.tv_ds, 4. / 7.)
            self.n_train, self.n_val, self.n_test = [len(ds) for ds in
                                                     [self.train_ds, self.val_ds, self.test_ds]]
            print("n_train, n_val, n_test: ", self.n_train, self.n_val, self.n_test)

            # Add the channel-wise normalization transform
            self.train_mean, self.train_std = self.train_ds.channel_mean[:self.n_channels], self.train_ds.channel_std[:self.n_channels]
            print("train channelwise_mean,std: ", self.train_mean, self.train_std)

            # Reset the image-transforms for each dataset based on training data's mean, std
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.in_shape[-2:]),
                transforms.Grayscale() if self.n_channels == 1 else torch.nn.Identity(),
                transforms.Normalize(mean=self.train_mean, std=self.train_std),

            ])
            self.train_ds.transform = train_transform

            if use_training_stat:
                self.val_ds.transform = train_transform
                self.test_ds.transform = train_transform

            # Reset each dataset's target_transform to be the same as the self.target_transform
            self.train_ds.target_transform = self.target_transform
            self.val_ds.target_transform = self.target_transform
            self.test_ds.transform = self.target_transform


    # return the dataloader for each split
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers)


    @staticmethod
    def unpack(batch: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        # delegate it to its Dataset object
        # consequently, any Dataset class must have unpack method (as static method)
        return MaptilesDataset.unpack(batch)


    @staticmethod
    def add_model_specific_args(parent_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
        # override existing arguments with new ones, if exists
        if parent_parser is not None:
            parents = [parent_parser]
        else:
            parents = []

        parser = ArgumentParser(parents=parents, add_help=False, conflict_handler='resolve')
        parser.add_argument('--data_root', type=str, default='/data/hayley-old/maptiles_v2/')
        parser.add_argument('--cities', nargs="+", type=str, default=["la", "paris"])
        parser.add_argument('--styles', nargs="+", type=str, default=["CartoVoyagerNoLabels", "StamenTonerBackground"])
        parser.add_argument('--zooms', nargs="+", type=str, default=["14"])

        parser.add_argument('--in_shape', nargs=3, type=int, default=[3, 32, 32])
        parser.add_argument('-bs', '--batch_size', type=int, default=32)
        parser.add_argument('--pin_memory', action="store_true", default=True)
        parser.add_argument('--num_workers', type=int, default=16)

        return parser