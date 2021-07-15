import pandas as pd
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable
from torch.utils.data import Dataset
from .two_factor_dataset import TwoFactorDataset
from src.visualize.utils import get_fig

class MaptilesDataset(TwoFactorDataset):

    _keys = ["img", "coord", "style"]
    _name_formatspec = "Maptiles_{cities_str}_{styles_str}_{zooms_str}"

    def __init__(self, *,
                 data_root: Path,
                 cities: Iterable[str],
                 styles: Iterable[str],
                 zooms: Iterable[str],
                 n_channels: int = 3,
                 transform: Optional[Callable] = None,
                 content_label_transform: Optional[Callable] = None,
                 style_label_transform: Optional[Callable] = None,

                 df_fns: pd.DataFrame = None,
                 verbose: bool = False):
        """
        If df_fns is given, no need to construct df_fns based on `data_root`, `cities`, `styles`, `zooms`.

        :param df_fns (pd.DataFrame): with columns = ['city', 'style', 'zoom', 'fpath']
        :param data_root:
        :param cities:
        :param styles:
        :param zooms:
        :param transform:
        :param target_transform:
        :param verbose:
        """
        self.data_root = data_root
        self.cities = sorted(cities)
        self.styles = sorted(styles)
        self.zooms = zooms
        self.n_channels = n_channels
        self.transform = transform
        self.content_label_transform = content_label_transform
        self.style_label_transform = style_label_transform

        if df_fns is not None:
            self.df_fns = df_fns
        else:
            self.df_fns = MaptilesDataset.collect_fns(self.data_root, self.cities, self.styles,
                                                      self.zooms, verbose=verbose)
        print("Unique styles: ", self.df_fns["style"].unique())

        self.channel_mean, self.channel_std = self.get_channelwise_mean_std(self, n_channels=self.n_channels)

        # style (str) to label (int) encoding
        self.style2idx = {s:i for i,s in enumerate(self.styles)}
        self.idx2style = {i:s for i,s in enumerate(self.styles)}

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, str, int]]:
        """
        Return `idx`th sample from the dataset

        -x: (np.ndarray) of 3dim H=256,W=256,C=3. Values are in range [0.,1.]
        -y (str): style name (long/original name)

        """
        *csz, fpath = self.df_fns.iloc[idx].to_list()
        try:
            x = plt.imread(fpath)[..., :3]
        except SyntaxError:  # read as jpg
            x = plt.imread(fpath, format='jpg')[..., :3]

        city, style, zoom = csz
        coord = fpath.stem.split("_")[:2] # geo-coordinates as (lng,lat)
        # label = {
        #     "city": city,
        #     "style": style,
        #     "zoom": zoom,
        #     "coord": coord
        # }

        if self.transform is not None:
            x = self.transform(x)
        if self.content_label_transform is not None:
            coord = self.content_label_transform(coord)
        if self.style_label_transform is not None:
            style = self.style_label_transform(style)

        return {"img": x, "coord": coord, "style": style}

    def __len__(self):
        "Return the number of samples in the dataset"
        return len(self.df_fns)

    def __repr__(self):
        return self.name

    @property
    def name(self) -> str:
        return self._name_formatspec.format(
            cities_str='-'.join(self.cities),
            styles_str='-'.join(self.styles),
            zooms_str='-'.join(self.zooms)
        )

    def make_subset(self,
                    inds: Iterable[int],
                    transform=None,
                    content_label_transform=None,
                    style_label_transform=None): # ->MaptilesDataset
        """
        Create a new MaptilesDataset object with a subset of df_fns
        and optionally overwritten transform and target_transform.

        :param inds:
        :param transform:
        :param target_transform:
        :return: MaptilesDataset
        """
        df_fns = self.df_fns.iloc[inds].reset_index(drop=True)
        return MaptilesDataset(
            data_root=self.data_root,
            cities=self.cities,
            styles=self.styles,
            zooms=self.zooms,
            n_channels=self.n_channels,
            transform=transform or self.transform,
            content_label_transform=content_label_transform or self.content_label_transform,
            style_label_transform=style_label_transform or self.style_label_transform,
            df_fns=df_fns,
        )

    def get_label_dict(self, idx):
        *csz, fpath = self.df_fns.iloc[idx].to_list()
        city, style, zoom = csz  # todo: what about city
        coord = fpath.stem.split("_")[:2]  # geo-coordinates as (lng,lat)
        label = {
            "city": city,
            "style": style,
            "zoom": zoom,
            "coord": coord
        }
        return label

    def get_coord(self, idx):
        return self.get_label_dict(idx)["coord"]

    def print_meta(self):
        print("Num. tiles: ", len(self))
        print("Cities: ", self.cities)
        print("Styles: ", self.styles)
        print("Zooms: ", self.zooms)

    def show_samples(self,
                     inds:Optional[Iterable[int]]=None,
                     n_samples:int=16,
                     order: str='hwc',
                     cmap=None):
        """
        :param inds:
        :param n_samples:

        If inds is not None, n_samples will be ignored.
        If inds is None, `n_samples` number of random samples will be shown.

        """
        if inds is None:
            inds = np.random.choice(len(self),n_samples,replace=False)

        f, axes = get_fig(n_samples)
        f.suptitle("Samples")
        for i, ax in enumerate(axes):
            ind = inds[i]
            batch = self[ind] #calls self.__getitem__
            img, _, _ = self.unpack(batch)
            label_dict = self.get_label_dict(ind)

            if order=="chw":
                img = img.permute(1,2,0)

            try:
                title = "-".join([label_dict["city"], label_dict["style"], label_dict["zoom"]])
                title = title + f"\n {label_dict['coord']}"
            except TypeError: #traget_transform must be not None
                title = label_dict

            if i < n_samples:
                ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                # ax.set_axis_off()
            else:
                f.delaxes(ax)

    @classmethod
    def get_counts(cls):
        return cls.df_fns.groupby(['city', 'style', 'zoom']).sum('fpath')

    @staticmethod
    def collect_fns(data_root: Path,
                    cities: Iterable[str] = None,
                    styles: Iterable[str] = None,
                    zooms: Iterable[str] = None,
                    verbose: bool = False,
                    ) -> pd.DataFrame:
        """Collect paths to all maptile files from `cities`, for each style in `styles`
        and at each zoom level in `zooms`.

        Parameters
        ----------
        data_root (Path): Path object to the root folder for data
        cities : Iterable[str]
            List of city names
        styles : Iterable[str]
            List of style names. Eg: ['OSMDefault', 'CartoVoyagerNoLabels']
        zooms : Iterable[str]
            List of zoom levels as string. Eg: ['14']  or ['14','15']

        Returns
        -------
        df_fns : pd.DataFrame with columns = ['city', 'style', 'zoom', 'fpath']
            a Pandas DataFrame containing maptile information for the queried
            criteria (in the Parameters). Each row contains a metadata about a
            maptile (its 'city', 'style', 'zoom', and 'fpath' (file path)

        TODO: Currently the `fn` column stores Path objects, rather than the string
             Is it better to store str objects?
        """
        # Collect as a record/row = Tuple[str, str, str, int] for a dataframe
        rows = []
        for city_dir in data_root.iterdir():
            if city_dir.is_dir():
                city = city_dir.name
                if verbose: print(f"\n{city}")
                if city not in cities:
                    if verbose: print(f"Skipping... {city}")
                    continue
                for style_dir in city_dir.iterdir():
                    if style_dir.is_dir():
                        style = style_dir.name
                        if verbose: print(f"\n\t{style}", end='...')
                        if style not in styles:
                            if verbose: print(f"Skipping... {style}")
                            continue
                        # breakpoint() #debug

                        for zoom_dir in style_dir.iterdir():
                            if zoom_dir.is_dir():
                                z = zoom_dir.name
                                if verbose: print(f"\n\t\t{z}")
                                if z not in zooms:
                                    if verbose: print(f"Skipping... {z}")
                                    continue
                                # # debug
                                # print(zoom_dir)
                                # breakpoint()
                                for fpath in zoom_dir.iterdir():
                                    if fpath.is_file():
                                        rows.append([city, style, z, fpath])

        # Construct a dataframe
        df_fns = pd.DataFrame(rows, columns=['city', 'style', 'zoom', 'fpath'])
        return df_fns

    @staticmethod
    def get_channelwise_mean_std(
            dset: Dataset,
            n_channels: int) -> Tuple[np.ndarray, np.ndarray]:
        """If unit8, we first convert to [0,1.0] range of float32.
        Modified: https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6
        """
        channel_sum = np.zeros(n_channels)
        channel_squared_sum = np.zeros(n_channels)
        n_pixels = 0.
        for i in range(len(dset)):
            fpath = dset.df_fns.iloc[i]["fpath"]
            try:
                img = plt.imread(fpath)[..., :n_channels]
            except SyntaxError:  # read as jpg
                img = plt.imread(fpath, format='jpg')[..., :n_channels]
            if img.dtype == np.uint8:
                img = img / np.float32(256.)

            n_pixels += img.size / n_channels
            channel_sum += np.sum(img, axis=(0, 1))
            channel_squared_sum += np.sum(img ** 2, axis=(0, 1))

        channel_mean = channel_sum / n_pixels
        channel_std = np.sqrt(channel_squared_sum / n_pixels - channel_mean ** 2)
        return channel_mean, channel_std

    @staticmethod
    def random_split(dset,
                     ratio: Union[float, Tuple[float, float]],
                     seed: int = None) -> Tuple[Dataset, Dataset]:
        """
        Split the indicies to a list of maptile file names into two groups by following the raio
        :param dset:
        :param ratio:
        :param seed:
        :return:
        """
        n = len(dset)
        r0 = ratio if isinstance(ratio, float) else ratio[0]
        n0 = math.ceil(n * r0)

        inds = np.arange(n)
        np.random.seed(seed)
        np.random.shuffle(inds)

        inds0, inds1 = inds[:n0], inds[n0:]
        return dset.make_subset(inds0), dset.make_subset(inds1)


class MapStyles():
    _long2short = {
        "EsriImagery": "Esri",
        "EsriWorldTopo": "EsriTopo",
        "CartoLightNoLabels": "CartoLight",
        "CartoVoyagerNoLabels": "CartoVoyager",
        "StamenTonerLines": "StamenTonerL",
        "StamenTonerBackground": "StamenTonerBg",
        "StamenTerrainLines": "StamenTerrainL",
        "StamenTerrainBackground": "StamenTerrainBg",
        "StamenWatercolor": "StamenWc",
        "OSMDefault": "OSM",
        "MtbmapDefault": " Mtb"
    }

    _long2id = {long:i for i,long in enumerate(_long2short.keys())}

    @classmethod
    def get_longnames(cls) -> List:
        return list(cls._long2short.keys())

    @classmethod
    def get_shortnames(cls) -> List:
        return list(cls._long2short.values())

    @classmethod
    def _short2long(cls):
        return {short: long for long, short in cls._long2short.items()}

    @classmethod
    def shortname(cls, style: str) -> str:
        return cls._long2short[style]

    @classmethod
    def longname(cls, short: str) -> str:
        return cls._short2long()[short]

    @classmethod
    def long2id(cls, long: str) -> int:
        return cls._long2id[long]

    # TODO: Implement as delegation; Add "remove" method
    @classmethod
    def update(cls, style: str, shortname: str) -> None:
        cls._long2short[style] = shortname

    def __init__(self):
        pass


# def test_mapstyles_long2short():
#     for s in styles:
#         print(f"{s}: {MapStyles.shortname(s)}")
#
#
# def test_mapstyles_short2long():
#     d = MapStyles._long2short
#     for long, short in d.items():
#         print(f"{short}: {MapStyles.longname(short)}")
#
#
# test_mapstyles_short2long()





