# must have attributes
# self.dims = tuple/list of C,H,W
# TODO:
# Make all the datamodule classes a child of this class
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable
import pytorch_lightning as pl

class BaseDataModule(pl.LightningDataModule):
    """Base DataModule. Any Datamodules will inherit from this base class.
    Specify extra contracts for defining a datamodule class that works with our experiment-run file, `train.py`
    on top of the pl.LightningDataModule's contract.

    Required init args:
    - data_root
    - in_shape
    - batch_size

    Optional init args:
    - pin_momery
    - num_workers
    - verbose

    Methods required to implement:
    - def name(self):

    Required attributes:
    - self.hparams

    """

    def __init__(self, *,
                 data_root: Union[Path, str],
                 in_shape: Tuple,
                 batch_size: int,
                 pin_memory: bool = True,
                 num_workers: int = 16,
                 shuffle: bool = True,
                 verbose: bool = False,
                 **kwargs):
        # required args
        super().__init__()
        self.data_root = data_root
        self.in_shape = in_shape
        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = in_shape


        # Training dataset's stat
        # Required to be set before being used in its Trainer
        self.train_mean, self.train_std = None, None

        # data loading
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.verbose = verbose

        # Keep main parameters for experiment logging
        self.hparams = {
            "in_shape": self.in_shape,
            "batch_size": self.batch_size
        }


    #todo: make it required
    @property
    def name(self) -> str:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, **kwargs):
        return cls(**kwargs)