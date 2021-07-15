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
from .multisource_maptiles_datamodule import MultiMaptilesDataModule


class MultiOSMnxRDataModule(MultiMaptilesDataModule):

    def __init__(
            self, *,
            data_root: Path,
            cities: Iterable[str],
            bgcolors: Iterable[str],
            zooms: Iterable[str],
            in_shape: Union[torch.Size, Tuple[int, int, int]],  # target image shape
            batch_size: int,

            transform: Callable = None,
            content_label_transform: Optional[Callable] = None,
            style_label_transform: Optional[Callable] = None,

            df_fns: pd.DataFrame = None,
            # --end of MaptilesDataset init args (except in_shape, batch_size)
            # Dataloading args
            pin_memory: bool = True,
            num_workers: int = 16,
            shuffle: bool = True,
            verbose: bool = False,
            **kwargs):
        """Same input signature as in MultiMaptilesDataModule except:
        - bgcolors (vs. styles): e.g. ['r','g','b','k']
        - edge_color (Optional; str): Default:'r'.
            Note: not a list.
        - lw_factor (Optional; float): Default: 0.5

        :param data_root:
        :param cities:
        :param styles: List of bgcolors (each is a string)
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
        # Set defaults for edge_color and lw_factor
        edge_color = kwargs.get('edge_color', 'cyan')
        lw_factor = kwargs.get('lw_factor', 0.5)
        bgcolors = sorted(bgcolors)

        # Set attributes specific to this subclass
        self.bgcolors = bgcolors
        self.edge_color = edge_color
        self.lw_factor = lw_factor

        styles = styles = [f'OSMnxR-{bgcolor}-{edge_color}-{lw_factor}' for bgcolor in bgcolors]

        super().__init__(
            data_root=data_root,
            cities=cities,
            styles=styles,
            zooms=zooms,
            in_shape=in_shape,
            batch_size=batch_size,
            transform=transform,
            content_label_transform=content_label_transform,
            style_label_transform=style_label_transform,
            df_fns=df_fns,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            verbose=verbose,
            **kwargs
        )

    @property
    def name(self):
        try:
            return self.train_ds.name
        except:
            return 'OSMnxRoads'


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_root', type=str, default="/data/hayley-old/osmnx_data/images")
        parser.add_argument('--cities', nargs="+", type=str, default=["la", "paris"])
        parser.add_argument('--bgcolors', nargs="+", type=str, default=['k', 'r', 'g', 'b', 'y'])
        parser.add_argument('--edge_color', type=str, default="cyan")
        parser.add_argument('--lw_factor', type=float, default=0.5)
        parser.add_argument('--zooms', nargs="+", type=str, default=["14"])

        parser.add_argument('--in_shape', nargs=3, type=int, default=[3, 32, 32])
        parser.add_argument('-bs', '--batch_size', type=int, default=32)
        parser.add_argument('--pin_memory', action="store_true", default=True)
        parser.add_argument('--num_workers', type=int, default=16)

        return parser