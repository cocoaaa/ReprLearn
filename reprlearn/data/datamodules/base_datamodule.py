# must have attributes
# self.dims = tuple/list of C,H,W
# TODO:
# Make all the datamodule classes a child of this class
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Any, Mapping, Union, Callable
import pytorch_lightning as pl
from argparse import ArgumentParser

class BaseDataModule(pl.LightningDataModule):
    """Base DataModule. Any Datamodules will inherit from this base class.
    Specify extra contracts for defining a datamodule class that works with our experiment-run file, `train.py`
    on top of the pl.LightningDataModule's contract.

    Required init args:
    - data_root
    - in_shape : size of a single datapoint tensor (e.g. (3,64,64) for a 64x64 RGB;
         Do not include the batch_size.
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
        # self.dims is returned when you call dm.size(), which is the size of
        # a single datapoint, e.g. (1,28,28)
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
        self.save_hyperparameters("in_shape", "batch_size")


    #todo: mark as required (e.g. @abc.abstractmethod context?)
    @property
    def name(self) -> str:
        """Name of this datamodule. Used e.g. for logging an experiment"""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, **kwargs):
        """Initialize a DataModule object from a config given as a dictionary"""
        return cls(**kwargs)

    #todo: mark as required (e.g. @abc.abstractmethod context?)
    @staticmethod
    def unpack(batch: Dict[str,Any]) -> Tuple:
        # delegate it to its Dataset object
        # consequently, any Dataset class must have unpack method (as static method)
        raise NotImplementedError

    @staticmethod
    def add_model_specific_args(parent_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
        # override existing arguments with new ones, if exists
        if parent_parser is not None:
            parents = [parent_parser]
        else:
            parents = []

        parser = ArgumentParser(parents=parents, add_help=False, conflict_handler='resolve')
        # todo: add arguments specific to this dataset module
        return parser