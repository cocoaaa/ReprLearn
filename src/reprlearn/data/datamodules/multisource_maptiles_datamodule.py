from pathlib import Path
from argparse import ArgumentParser
from typing import List, Set, Dict, Tuple, Optional, Iterable, Any, Union, Callable
from collections import defaultdict
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl

from src.data.datasets.maptiles_bifactor import MaptilesDataset
from .multisource_datamodule import MultiSourceDataModule

class MultiMaptilesDataModule(MultiSourceDataModule):

    # _name_formatspec = MaptilesDataset._name_formatspec

    def __init__(self, *,
                 data_root: Path,
                 cities: Iterable[str],
                 styles: Iterable[str],
                 zooms: Iterable[str],
                 in_shape: Union[torch.Size, Tuple[int, int, int]],  # target image shape
                 batch_size: int,

                 transform: Callable = None,
                 content_label_transform: Optional[Callable] = None,
                 style_label_transform: Optional[Callable] = None,

                 df_fns: pd.DataFrame = None,
                 # --end of MaptilesDataset init args (except in_shape, batch_size)
                 # Dataloading args
                 pin_memory: bool=True,
                 num_workers: int = 16,
                 shuffle: bool = True,
                 verbose: bool = False,
                 **kwargs):
        """

        :param data_root:
        :param cities:
        :param styles:
        :param zooms:
        :param transform:
        :param content_label_transform:
        :param style_label_transform:
        :param df_fns:
        :param verbose:
        :param in_shape:
        :param batch_size:
        :param pin_memory:
        :param num_workers:
        :param shuffle:
        :param kwargs:
        """
        styles = sorted(styles)
        super().__init__(
            data_root=data_root,
            n_contents=len(cities),
            source_names=styles,
            in_shape=in_shape,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            verbose=verbose,
            **kwargs
        )

        # Set attributes specific to this dataset
        self.cities = cities
        self.styles = styles
        self.zooms = zooms
        self.df_fns = df_fns

        # self.n_contents = len(self.cities) #set by super class
        self.n_styles = len(self.styles)

        # Style string <-> style_idx
        self.style2idx = {s:i for i,s in enumerate(self.styles)}
        self.idx2style = {i:s for i,s in enumerate(self.styles)}

        # transforms
        self.transform = transform
        self.content_label_transform = content_label_transform
        self.style_label_transform = style_label_transform
        self.n_channels, self.in_h, self.in_w = in_shape

        # default transforms
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.in_shape[-2:]),
            ])

        if self.content_label_transform is None:
            self.content_label_transform = transforms.Lambda(
                    lambda lnglat_list: '-'.join(lnglat_list)
            )
        if self.style_label_transform is None:
            self.style_label_transform = transforms.Lambda(
                    lambda style_label: self.style2idx[style_label]
            )

        # Update hparams with maptiles specifics
        self.hparams.update({
            "cities":  self.cities,
            "styles": self.styles,
            "zooms": self.zooms,
            "n_styles": self.n_styles,
         })


    @classmethod
    def from_maptiles_dataset(cls, dset: MaptilesDataset, *,
                              in_shape: Tuple[int, int, int],
                              batch_size: int = 32,
                              pin_memory: bool = True,
                              num_workers: int = 16,
                              verbose: bool = False,
                              ):
        kwargs = {
            'data_root': dset.data_root,
            'cities': dset.cities,
            'styles': dset.styles,
            'zooms': dset.zooms,
            'transform': dset.transform,
            'content_label_transform': dset.content_label_transform,
            'style_label_transform': dset.style_label_transform,
            # --end of MaptilesDataset init args
            'in_shape': in_shape,
            'batch_size': batch_size,
            'verbose': verbose,
            'pin_memory': pin_memory,
            'num_workers': num_workers,
        }
        return cls.from_dict(**kwargs)

    @classmethod
    def unpack(cls, batch: Dict[str, Any]) -> Tuple[Any]:
        """Unpacks a batch as a dictionary to a tuple of (data_x, content_label, style_label),
        so that the dataloading implementation is similar to standard torch's dataset objects
        In every Multisource dataset, we delegate it to each homogeneous dataset class.

        Parameters
        ----------
        batch : Dict[str,Any]
            a sample from the dataset returned by self.__getitem__(), containing the data("x"),
            content-label and style-label

        Returns
        -------
        a tuple of the data, its content label and its style label
        """
        return MaptilesDataset.unpack(batch)

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
        """
        This function is called on every GPU in a node/machine
        Sets self.train_ds, self.val_ds
        -- this also configures this DataModule to have a specified transforms
        -- that will be applied to each sample in the dataset
        """
        # breakpoint()
        dset = MaptilesDataset(
            df_fns=self.df_fns,
            data_root=self.data_root,
            cities=self.cities,
            styles=self.styles,
            zooms=self.zooms,
            transform=self.transform,
            content_label_transform=self.content_label_transform,
            style_label_transform=self.style_label_transform,
            verbose=self.verbose)
        if self.df_fns is None:
            self.df_fns = dset.df_fns
            print("*** Set the datamodule's df_fns attribute -- Save it for quicker DM init for later runs")

        # split to train/val or test
        if stage == 'fit':
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

    # return the dataloader for each split
    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers,
                          drop_last=True)

    def get_content_samples(self,
                            n_unique_contents: int,
                            mode='train'
                            ) -> Dict[str, torch.Tensor]:
        content_samples = {}
        ds = getattr(self, f"{mode}_ds")

        # def has_collected_all():
        #     if len(samples_per_content) < n_unique_contents:
        #         return False
        #
        #     for content_id, list_timgs in samples_per_content.items():
        #         if len(list_timgs) < self.n_styles:
        #             return False
        #
        #     return True

        for i in range(len(ds)):
            if len(content_samples) >= n_unique_contents:
                return content_samples
            batch = ds[i]
            x, label_c, label_s = self.unpack(batch)
            label_c_str = '-'.join(label_c)
            content_samples[label_c_str] = x
        return content_samples

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_root', type=str, default='/data/hayley-old/maptiles_v2/')
        parser.add_argument('--cities', nargs="+", type=str, default=["la", "paris"])
        parser.add_argument('--styles', nargs="+", type=str, default=["CartoVoyagerNoLabels", "StamenTonerBackground"])
        parser.add_argument('--zooms', nargs="+", type=str, default=["14"])

        parser.add_argument('--in_shape', nargs=3, type=int, default=[3, 32, 32])
        parser.add_argument('-bs', '--batch_size', type=int, default=32)
        parser.add_argument('--pin_memory', action="store_true", default=True)
        parser.add_argument('--num_workers', type=int, default=16)

        return parser