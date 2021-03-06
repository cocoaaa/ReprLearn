import pandas as pd
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Any, Union, Callable
from torch.utils.data import Dataset

from reprlearn.visualize.utils import get_fig
#debug
from IPython.core.debugger import  set_trace

class MaptilesDataset(Dataset):
    """
    If df_fns is given, no need to construct df_fns based on `data_root`, `cities`, `styles`, `zooms`.

    :param df_fns (pd.DataFrame): with columns = ['city', 'style', 'zoom', 'fpath']
    :param data_root: e.g. Path("/data/hayley-old/maptiles_v2/")
    :param cities: .e.g ['la', 'paris', 'london']
    :param styles:
    :param zooms:
    :param transform:
    :param target_transform:
    :param verbose:
    """
    # class attributes
    _name_formatspec = "Maptiles_{cities_str}_{styles_str}_{zooms_str}"

    def __init__(self, *,
                 data_root: Path,
                 cities: Iterable,
                 styles: Iterable,
                 zooms: Iterable[str],
                 n_channels: int = 3,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 df_fns: pd.DataFrame = None,
                 verbose: bool = False):

        self.cities = cities
        self.styles = sorted(styles)
        self.zooms = zooms
        self.n_channels = n_channels
        self.transform = transform
        self.target_transform = target_transform

        self.data_root = data_root

        if df_fns is not None:
            self.df_fns = df_fns
        else:
            self.df_fns = MaptilesDataset.collect_fns(self.data_root, self.cities, self.styles,
                                                      self.zooms, verbose=verbose)
        print("Unique styles: ", self.df_fns["style"].unique())

        self.channel_mean, self.channel_std = self.get_channelwise_mean_std(self, n_channels=self.n_channels)

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.df_fns)

    def __repr__(self):
        return self.name

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img, metadata = self.read_data(index)

        #todo: what 'y' and 'd' mean should be flexible..
        return {
            'x': img,
            # 'y': np.array([int(coord_i) for coord_i in metadata['coord']]), #np.array([int(lng), int(lat)])
            'y': metadata['city'], #todo: perhaps make a string f'{metadata['coord'][0]}-{metadata['coord'][1]}' which is 'lng-lat'
            'd': metadata['style']
        }

    def read_data(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Helper method to read image (maptile) and metadata from disk
        Return `idx`th sample from the dataset as a dictionary
        - 'x' : (np.ndarray) of 3dim H=256,W=256,C=3. Values are in range [0.,1.]
        - 'y' : (str) style name (long/original name)
        """
        *csz, fpath = self.df_fns.iloc[index].to_list()
        try:
            img = plt.imread(fpath)[..., :3] #remove alpha channel
        except SyntaxError:  # read as jpg
            img = plt.imread(fpath, format='jpg')[..., :3]

        city, style, zoom = csz
        coord = np.array(fpath.stem.split("_")[:2]) # geo-coordinates as (lng,lat)
        metadata = {
            "city": city,
            "style": style,
            "zoom": zoom,
            "coord": coord
        }

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            metadata = self.target_transform(metadata)

        return img, metadata

    @staticmethod
    def unpack(batch: Dict[str,Any]) -> Tuple:
        # delegate it to its Dataset object
        # consequently, any Dataset class must have unpack method (as static method)
        return batch['x'], batch['y'], batch['d']

    @property
    def name(self) -> str:
        return self._name_formatspec.format(
            cities_str='-'.join(self.cities),
            styles_str='-'.join(self.styles),
            zooms_str='-'.join(self.zooms)
        )

    def make_subset(self, inds: Iterable[int],
                    transform=None,
                    target_transform=None
                    ):
        """
        Create a new MaptilesDataset object with a subset of df_fns
        and optionally overwritten transform and target_transform.

        :param inds:
        :param transform:
        :param target_transform:
        :return:
        """
        df_fns = self.df_fns.iloc[inds].reset_index(drop=True)
        return MaptilesDataset(
            data_root=self.data_root,
            cities=self.cities,
           styles=self.styles,
           zooms=self.zooms,
           n_channels=self.n_channels,
           transform=transform if transform is not None else self.transform,
           target_transform=target_transform if target_transform is not None else self.target_transform,
           df_fns=df_fns
        )

    def get_label(self, idx):
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
        return self.get_label(idx)["coord"]

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
            img, label = self[ind] #calls self.__getitem__
            if order=="chw":
                img = img.permute(1,2,0)

            try:
                title = "-".join([label["city"], label["style"], label["zoom"]])
                title = title + f"\n {label['coord']}"
            except TypeError: #traget_transform must be not None
                title = label

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
        """
        Collect all Count the number of maptiles from `cities`, for each style in `styles`
        and at each zoom level in `zooms`

        Args:
        - data_root (Path): Path object to the root folder for data


        - debug (bool)
        - n_show (int): number of images to sample and show for each city/style/zoom

        Note: If debug is false, n_show is ignored

        Returns:
        - fns (pd.DataFrame): with columns = ['city', 'style', 'zoom', 'fpath']

        TODO: the `fn` column stores Path objects (rather than the string)?
        -- or better to store str object?
        """
        # Collect as a record/row = Tuple[str, str, str, int] for a dataframe
        rows = []
        for city_dir in data_root.iterdir():
            if city_dir.is_dir():
                city = city_dir.stem
                if verbose: print(f"\n{city}")
                if city not in cities:
                    if verbose: print(f"Skipping... {city}")
                    continue
                for style_dir in city_dir.iterdir():
                    if style_dir.is_dir():
                        style = style_dir.stem
                        if verbose: print(f"\n\t{style}")
                        if style not in styles:
                            if verbose: print(f"Skipping... {style}")
                            continue
                        for zoom_dir in style_dir.iterdir():
                            if zoom_dir.is_dir():
                                z = zoom_dir.stem
                                if verbose: print(f"\n\t\t{z}")
                                if z not in zooms:
                                    if verbose: print(f"Skipping... {z}")
                                    continue
                                for fpath in zoom_dir.iterdir():
                                    if fpath.is_file():
                                        rows.append([city, style, z, fpath])

        # Construct a dataframe
        df_counts = pd.DataFrame(rows, columns=['city', 'style', 'zoom', 'fpath'])
        return df_counts

    @staticmethod
    def get_channelwise_mean_std(
            dset: Dataset,
            n_channels: int) -> Tuple[np.ndarray, np.ndarray]:
        """ Modified: https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6

        Assumes
         batch = dset[ind]
         x = batch['x'] # torch or np image of  float dtype , not uint8 image
         # x is in range(0.0, 1.0)

         Returns
         -------
         channel_mean : (np.ndarray of len = num_channels)
         channel_std : (np.ndarray of length `nC`)
        """
        channel_sum = np.zeros(n_channels)
        channel_squared_sum = np.zeros(n_channels)
        n_pixels_per_channel = 0.
        for i in range(len(dset)):
            img = dset[i]['x']
            if isinstance(img, torch.Tensor): # flip order from (c,h,w) to (h,w,nc)
                img = img.permute(1,2,0).numpy()
            n_pixels_per_channel += img.size / n_channels
            channel_sum += np.sum(img, axis=(0, 1))
            channel_squared_sum += np.sum(img ** 2, axis=(0, 1))
        channel_mean = channel_sum / n_pixels_per_channel
        channel_std = np.sqrt(channel_squared_sum / n_pixels_per_channel - channel_mean ** 2)
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


class MapCities():
    _cities = [
        "amsterdam", "berlin", "boston",
        "charlotte", "chicago", "la",
        "london", "manhattan", "montreal",
        "paris", "rome", "seoul",
        "shanghai", "vegas"
    ]

    @classmethod
    def get_names(cls) -> List:
        return cls._cities




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





