from pathlib import Path
from typing import Tuple, Iterable, Optional, Callable, List, Dict, Union
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torchvision.datasets import VisionDataset

class DigitSumDataset(VisionDataset):
    """Each datapoint is a Tuple[List[image], int]],
    where the target variable is the sum of the digits in the List of images.
    
    Note: each call to __getitem__ is random, so two seperate calls to `dset[ind]` 
    with the same ind will result in different random samples.
    - todo: [ ] maybe i should name this class "RandomDigitsumDataset" 
    
    
    Args
    -----
    data_x : Tensor; (n_data, *data_shape)
        a tensor containing all available datapt x's
    data_y : Tensor; (n_data, *target_shape)
        a tensor containing correpsonding lables of datapt x's
    dset_len : int
        number of datapoints in one epoch 
        where a datapoint of this dataset is a tuple of (setx,y)
    max_set_size : int
        maximum length among any set_x in this dataset
    fix_set_size : bool
        If true, any set_x in this dataset has the same size of `max_set_size`
        Else, each call to __getitem__ randomly samples a set_size in [1, max_set_size]
    seed : int
        random seed fixed at the initialization of the Dataset instance
    x_transform : Optional[Callable]
        transform applied to each element in a set_x to get:
        `set_transformedx = [ x_transform(x) for x in set_x]`
        Note that x_transform should work on the input of a Tensor. (not PIL or ndarray)
    target_tranform : Optional[Callable]
        tranform applied at the sum_value y
        
        
    Returns 
    -------
    Each item returned is a tuple of set_x and set_y:
    - set_x : Tensor of shape (set_size, nC, h, w)
    - y: Tensor of integer type, correpsonding to the sum of digits represented in set_x
    
    """
    def __init__(self, 
                 data_x: Tensor, 
                 data_y: Tensor, 
                 dset_len: int,
                 max_set_size: int, 
                 fix_set_size: bool=True, 
                 x_transform: Optional[Callable]=None,
                 target_transform: Optional[Callable]=None,
                 target_dtype: Optional[Union[type, torch.dtype]]=torch.float32,
                 seed: Optional[int]=None,
                 ):
        
        self.data_x = data_x
        self.data_y = data_y
        self.n_data = len(data_x)
        self.dset_len = dset_len
        self.max_set_size = max_set_size
        self.fix_set_size = fix_set_size
        self.x_transform = x_transform
        self.target_transform = target_transform
        self.target_dtype = target_dtype
        self.seed = seed # random seed for each image sampling in __getitem__
        # set the random seed so that the sequence of calls to __getitem__ results in 
        # the predictable sequence of `set_size` to be sampled 
        if self.seed is not None:
            np.random.seed(self.seed)
            
        
    def __len__(self):
        # number of sets/sequences of digit-images we want this dataset to contain  
        # in one epoch of the dataset
        # return np.iinfo(int).max  #todo: hacky; ideally, we should make use of itertools.cycle
        return self.dset_len
    
    def __getitem__(self, ind) -> Tuple[Iterable[Tensor], Iterable[Tensor]]:
        """
        Args
        ----
        ind : int
            It does not have any meaningful effect in this implementation. 
            In particular, this `ind` has no relation to the indices of `self.data_x` 
            or self.data_y`

        Returns
        -------
        Tuple of set_x, set_y
        """
        if self.fix_set_size:
            set_size = self.max_set_size
        else:
            set_size = np.random.randint(1, self.max_set_size+1)
            # print('set_size: ', set_size)
        # select `set_size` number of indices in range(0, self.n_data)
        dpt_inds = np.random.randint(self.n_data, size=set_size)
        
        # set_x: input variable
        set_x = self.data_x[dpt_inds] #tensor as the same size of the original tensor data_x
        if self.x_transform is not None:
            set_x = torch.stack([self.x_transform(x) for x in set_x ])
            # print('set_x shape after transform: ', set_x.shape)                    
        
        # target variable  (sum of the elements in set_x)                     
        set_y = self.data_y[dpt_inds]
        y = set_y.sum(dtype=self.target_dtype).reshape((1,1)) #torch.float32
        if self.target_transform is not None:
            y = self.target_transform(y)
        
        return (set_x, y)
        
        