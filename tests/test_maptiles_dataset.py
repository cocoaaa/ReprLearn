from pathlib import Path
from src.data.datasets.maptiles import MaptilesDataset
from src.data.datamodules.maptiles_datamodule import MaptilesDataModule

DATA_ROOT = Path("/data/hayley-old/maptiles_v2/")

def test_maptiles_dataset():
    cities = ['paris']
    styles = ['CartoVoyagerNoLabels', 'StamenTonerBackground']
    zooms = ['15']
    n_channels = 3
    dset = MaptilesDataset(
        data_root=DATA_ROOT,
        cities=cities,
        styles=styles,
        zooms=zooms,
        n_channels=n_channels
    )
    train_dset, val_dset = MaptilesDataset.random_split(dset, 0.9)
    train_channel_mean, train_channel_std = train_dset.channel_mean, train_dset.channel_std

    [print(len(x)) for x in [dset, train_dset, val_dset]];
    [print(x.channel_mean, x.channel_std) for x in [dset, train_dset, val_dset]];


def test_maptiles_datamodule_factory():
    cities = ['paris']
    styles = ['CartoVoyagerNoLabels', 'StamenTonerBackground']
    zooms = ['15']
    n_channels = 3
    dset = MaptilesDataset(
        data_root=DATA_ROOT,
        cities=cities,
        styles=styles,
        zooms=zooms,
        n_channels=n_channels
    )

    in_shape = (n_channels, 64, 64)
    dm = MaptilesDataModule.from_maptiles_dataset(dset, in_shape=in_shape)
    dm.setup('test')
    print(dm.train_ds.channel_mean, dm.train_ds.channel_std)
    print(dm.train_ds.transform)
    print(dm.val_ds.transform)
    print(dm.test_ds.transform)