# must have attributes
# self.dims = tuple/list of C,H,W
# TODO:
# Make all the datamodule classes a child of this class
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset


class MultiSourceDataModule(pl.LightningDataModule):
    """
    todo:
    Consider: rename it to TwoFactorDataModule (in alignment with `src.data.datasets.two_factor_dataset.py`
    pros:
        the name "two-factor" is general so that we can swap the definition of content-class (y) and the style-class(d)
        if needed to experiment with different definitions of "content" and "style"
        E.g.: in Rotated MNIST, one way to define "content" vs. "style" is: `content-class` as digit-id vs. `style_class` as rotation
             Yet another way to define them is: content-class as rotation, and style-class as digit-id.
        Calling this class as "TwoFactor", rather than "MultiSource" makes it clear that the problem we are concerned with
        are about the (two) underlying factors of variations, rather than phrasing the (same/similar) problem as concerned with
        data that is consisted of multi-types/sources of data sets.

    cons:
        when we phrase the problem at hand as multi-source integration or domain(source)-independent representation learning, e.g.,
        then describing this datamodule with a word "MultiSource" elucidates that point clearly in its name.

    decision:
        whether which name of this DataModule more clearly describes what kind of data should be rather independent on
        the application/the ML problem we will use this datamodule for training a model.
        this independence is helpful because, then, we can use this DataModule in different kinds of applications (e.g. from the
        perspective of factor-ization of perceptural/sensory observations like in Tenenbaum2000 or JimGlass's paper2019,
        or, from the perspective of multi-source data integration or repr.learning of source/domain-invariant representation)
        so....?
    end

    BaseDataModule for a datamodule containing both content and style lables, aka. Multisource DataModules
    Each "source" is considered a "style".
    Outputted data samples are supervised, ie. a tuple of (x, y), eg. (batch_imgs, batch_content_labels)
    - Therefore, we have two kinds of labels, one for the style label and another for the content label

    Specify extra contracts for defining a datamodule class that works with our experiment-run file, `train.py`
    on top of the pl.LightningDataModule's contract:

    Required init args:
    - data_root
    - n_contents: num of content labels. Eg. 10 for MNIST datasets
    - source_names: list of sources (ie. styles)
        - Eg. ["Red", "Green"] for ConcatDataset(RedMNIST, GreenMNIST)
    - in_shape
    - batch_size

    Optional init args:
    - pin_memory
    - num_workers
    - verbose

    Methods required to implement:
    - def name(self):

    Required attributes:
    - self.hparams

    """
    def __init__(self, *,
                 data_root: Path,
                 n_contents: int,
                 source_names: List[str],
                 in_shape: Tuple,
                 # Dataloading args
                 batch_size: int,
                 pin_memory: bool = True,
                 num_workers: int = 16,
                 shuffle: bool = True,
                 test_shuffle:  bool = False,
                 verbose: bool = False,
                 **kwargs):
        # required args
        super().__init__()
        # Full dataset that concatenates multiple datasets
        self.data_root = data_root
        self.n_contents = n_contents
        self.source_names = source_names
        self.n_styles = len(source_names)
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
        self.test_shuffle = test_shuffle
        self.verbose = verbose

        # Keep main parameters for experiment logging
        self.hparams = {
            "n_contents": self.n_contents,
            "n_styles": self.n_styles,
            "source_names": self.source_names,
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

    def get_content_style_reps(self, mode: str) -> np.ndarray:
        """Returns a 2d matrix whose rows are content-labels, and cols are style-labels
        e.g. out[content_id, style_id] is an tensor image with digit0, style(angle)=0

        todo: make a class of "reps" of each dataset (e.g. MNISTr)
        - implement this function as a method
        - add a function that grabs the image of content_label i and style_label j
        """
        h, w = self.in_shape[-2:]
        dl = getattr(self, f'{mode}_dataloader')()
        ds = dl.dataset

        #todo: replace the following with TwoFactorDataset's `get_content_style_reps` method:
        #return ds.get_content_style_reps(self.n_contents, self.n_styles)
        reps = np.zeros((self.n_styles * h, self.n_contents * w))
        is_collected = np.zeros((self.n_styles, self.n_contents))
        for i in range(len(ds)):
            if is_collected.all():
                break

            sample = ds[i]
            x, y, d = self.unpack(sample)
            #         if isinstance(label_c, torch.Tensor):
            #             label_c = label_c.item()
            reps[h*d:h*(d+1), w*y:w*(y+1)] = x.numpy()
            is_collected[d, y] = True

        return reps

    def select_a_content_style_repr(
            self,
            reps: np.ndarray,
            d: int,  # style-label (index into row)
            y: int,  # content-label (index into col)
    ) -> np.ndarray:
        """Given the 2-dim (if each image is grayscale; if color 3-dim) np.array that is
        a collection of representative/random sample image of each contnet (row) and each style (col),
        return the image as np.array of the sample image of content_label=y and style_label=d

        args
        ----
            reprs (nd.ndarray) : image of style_label d and content_label y is stored at reprs
            d (int) : style class label
            y (int) : content class label

        returns
        -------
            2 or 3 dim np.ndarray for the image representative of style d and content y
        """
        if d >= self.n_styles:
            raise ValueError(f"d must be in range({self.n_styles})", d)
        if y >= self.n_contents:
            raise ValueError(f"y must be in range({self.n_contents})", y)

        h, w = self.in_shape[-2:]
        return reps[h*d:h*(d+1), w*y:w*(y+1)]

