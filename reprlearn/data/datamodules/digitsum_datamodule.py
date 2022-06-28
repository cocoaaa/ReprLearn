# lightening datamodule
from typing import Any,List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar
from pathlib import Path

from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, random_split

from pytorch_lightning import LightningDataModule

from reprlearn.data.datasets.digitsum import DigitSumDataset

class DigitsumDatamodule(LightningDataModule):
    def __init__(self, data_root: Path,
                  max_set_size: int, 
                 fix_set_size: bool, 
                  batch_size: int,
                 n_sets_per_train_epoch: int,
                 n_sets_per_val_epoch: int,
                 n_sets_per_test_epoch: int,
                 x_transform: Optional[Callable]=None,
                 target_transform: Optional[Callable]=None,
                 seed: Optional[int]=None,
                pin_memory: bool = True,
                 num_workers: int = 0,
                 shuffle: bool = True,
                 verbose: bool = False,
                 **kwargs
                ):
        self.data_root = data_root 
        self.n_sets_per_train_epoch = n_sets_per_train_epoch
        self.n_sets_per_val_epoch = n_sets_per_val_epoch
        self.n_sets_per_test_epoch = n_sets_per_test_epoch
        self.n_sets_per_predict_epoch = n_sets_per_test_epoch
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.verbose = verbose
        
        self.dset_args = {
            'max_set_size': max_set_size,
            'fix_set_size': fix_set_size,
            'x_transform': x_transform,
            'target_transform': target_transform,
            'seed': seed,
            
        }
        
    @property
    def name(self):
        return "DigitSumDataModule"
        
    def setup(self, stage: Optional[str] = None):
        # original mnist data
        mnist_test = MNIST(self.data_root, train=False)
        mnist_predict = MNIST(self.data_root, train=False)
        mnist_full = MNIST(self.data_root, train=True)
        mnist_train, mnist_val = random_split(mnist_full, [55000, 5000]) 
        
        # digitsum dataset, created from each split of mnist dataset
        self.train_dset = DigitSumDataset(
            data_x = mnist_full.data[mnist_train.indices, None, ...],
            data_y = mnist_full.targets[mnist_train.indices],
            dset_len = self.n_sets_per_train_epoch,
            **self.dset_args
        )
        
        self.val_dset =  DigitSumDataset(
            data_x = mnist_full.data[mnist_val.indices, None, ...],
            data_y = mnist_full.targets[mnist_val.indices],
            dset_len = self.n_sets_per_val_epoch,
            **self.dset_args
        )
        self.test_dset =  DigitSumDataset(
            data_x = mnist_test.data[:, None, ...],
            data_y = mnist_test.targets,
            dset_len = self.n_sets_per_test_epoch,
            **self.dset_args
        )
        self.predict_dset =  DigitSumDataset(
            data_x = mnist_predict.data[:, None, ...],
            data_y = mnist_predict.targets,
            dset_len = self.n_sets_per_predict_epoch,
            **self.dset_args
        )
        
#         print("mnist Dataset's underlying dataset shape: ",
#               mnist_full.data[:,None,...].shape, 
#               mnist_full.targets.shape)
#         print("same as the derivitives (train/val/test/predict_dset)'s underlying dataset shape: ",
#               self.train_dset.data_x.shape,
#               self.train_dset.data_y.shape
#              )
#         breakpoint() 
    
    def train_dataloader(self): #todo: fix
        return DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=self.shuffle,
                         pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self): #todo: fix
        return DataLoader(self.val_dset, batch_size=self.batch_size,
                         pin_memory=self.pin_memory, num_workers=self.num_workers)
                        

    def test_dataloader(self):
        return DataLoader(self.test_dset, batch_size=self.batch_size,
                         pin_memory=self.pin_memory, num_workers=self.num_workers)
                         

    def predict_dataloader(self):
        return DataLoader(self.predict_dset, batch_size=self.batch_size,
                         pin_memory=self.pin_memory, num_workers=self.num_workers)
                          

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        pass
        
                
# dl_digitsum