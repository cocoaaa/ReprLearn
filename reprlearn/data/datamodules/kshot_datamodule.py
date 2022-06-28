from typing import Callable, List, Optional
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pytorch_lightning as pl

from reprlearn.data.datasets.kshot_dataset import KShotImageDataset
from reprlearn.data.dataloaders.kshot_dataloader import KShotDataLoader

class KShotDataModule(pl.LightningDataModule):

    def __init__(self,
                 dataset_name: str,
                 data_root: Path,
                 k_shot: int,
                 n_way: int,
                 num_tasks_per_iter_for_train: int,
                 num_tasks_per_iter_for_eval: int,
                 max_iter_for_train: int,
                 max_iter_for_eval: Optional[int]=None,
                 train_mean: Optional[List]=None,
                 train_std: Optional[List]=None,
                 **kwargs
                 ):
        """
        dataset_name : str
          default is 'cifar100'. Currently only cifar100 is supported
        k_shot : int
        n_way : int
        num_tasks_per_iter : int
          number of task-sets in a batch from a kshot-dataloader
        train_mean/std : torch.Tensor
          channel-wise mean/std of train dataset. Eg.  [0.50707516 0.48654887 0.44091784]
        kwargs : Dict[str,Any]
          kwargs to be passed to KShotDataLoader
          shuffle : bool. default is True
          collate_fn : Callable. default is torch.stack
          --- todo: remove these two below --->
          pin_memory : bool. default is True
            used as `pin_memory` for all train/val/dataloaders
          num_workers : int. default is 2.
            used as `num_workers` for all train/val/dataloaders
          <--- end todo ---
        """
        super().__init__()
        self.dataset_name = dataset_name.upper()
        if self.dataset_name == 'CIFAR100':
            self.dataset_n_classes = 100
        else:
            self.dataset_n_classes = None
        self.data_root = data_root
        self.k_shot = k_shot
        self.n_way = n_way
        self.num_tasks_per_iter_for_train = num_tasks_per_iter_for_train #aka. batch_size for KShotDataLoader
        self.max_iter_for_train = max_iter_for_train
        self.num_tasks_per_iter_for_val = num_tasks_per_iter_for_eval
        self.num_tasks_per_iter_for_test = num_tasks_per_iter_for_eval
        self.max_iter_for_val = max_iter_for_eval
        self.max_iter_for_test = max_iter_for_eval

        self.shared_dl_config = {
            'k_shot': self.k_shot,
            'n_way': self.n_way,
            # 'pin_memory': True,
            # 'num_workers': 2,
        }
        self.shared_dl_config.update(kwargs)

        # train dataset's channelwise mean and std (for faster creation of the datamodule instance)
        self.mean_train_data = train_mean
        self.std_train_data = train_std


    @property
    def name(self) -> str:
        return f"{self.dataset_name}-{self.k_shot}shot-{self.n_way}way"

    def prepare_data(self):
        # download, split, etc
        # todo: get cifar100
        dset_train = datasets.CIFAR100(root=self.data_root, train=True, download=True,
                                       transform=transforms.ToTensor())
        dset_test = datasets.CIFAR100(root=self.data_root, train=False, download=True,
                                      transform=transforms.ToTensor())
        self.idx2str = {v: k for k, v in dset_train.class_to_idx.items()}

        # concatenate train and test cifar datasets and split them by classes
        # for train/val/test datasets for kshot-classifcation set up
        # -- dset_train.data: contains 50k np.ndarrays (32,32,3)
        # -- dset_train.targets: contains 50k class labels as List[int]
        all_imgs = np.concatenate((dset_train.data, dset_test.data), axis=0)  # (60k, 32, 32, 3) np.ndarray
        all_targets = torch.LongTensor(dset_train.targets + dset_test.targets)  # tensor of length 60k
        print(all_imgs.shape, all_targets.shape)

        # partition 100 classes in cifar1000 into 3 groups
        inds = torch.randperm(self.dataset_n_classes)
        # todo: fix hard-coded 80:10:10 split to ratio
        train_classes, val_classes, test_classes = inds[:80], inds[80:90], inds[90:]
        print("train classes: ", [self.idx2str[i.item()] for i in train_classes[:5]])
        print("val classes: ", [self.idx2str[i.item()] for i in val_classes[:5]])
        print("test classes: ", [self.idx2str[i.item()] for i in test_classes[:5]])

        # ---> todo: move to setup?
        if self.mean_train_data is None:
            self.mean_train_data = (dset_train.data / 255.).mean(axis=(0, 1, 2))  # (N, h,w,c) np.ndarray
        if self.std_train_data is None:
            self.std_train_data = (dset_train.data / 255.).std(axis=(0, 1, 2))
        # mean_train_data = self.mean_train_data or (dset_train.data / 255.).mean(axis=(0, 1, 2))  # (N, h,w,c) np.ndarray
        # std_train_data = self.std_train_data or (dset_train.data / 255.).std(axis=(0, 1, 2))
        print("cifar100 train: channel_mean: ", self.mean_train_data)
        print("cifar100 train: channel_sted: ", self.std_train_data)

        # Define transforms for meta-train and meta-val/test datasets
        # We assume inputs to these transforms are np.ndarray

        test_xform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean_train_data, self.std_train_data)
            ]
        )

        # for training dataset, additionally use data augmentation
        train_xform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean_train_data, self.std_train_data)
            ]
        )

        # create meta-train, meta-val, meta-test sets
        self.meta_train_dset = create_dset_from_classes(all_imgs, all_targets, train_classes, img_xform=train_xform)
        self.meta_val_dset = create_dset_from_classes(all_imgs, all_targets, val_classes, img_xform=test_xform)
        self.meta_test_dset = create_dset_from_classes(all_imgs, all_targets, test_classes, img_xform=test_xform)

        # set the self.max_iter_for_val and self.max_iter_for_test so that
        # epoch-wise val/test metric and loss is an average over the full val/test dataset
        self.max_iter_for_val = self.max_iter_for_val or len(self.meta_val_dset) // self.num_tasks_per_iter_for_val
        self.max_iter_for_test = self.max_iter_for_test or len(self.meta_test_dset) // self.num_tasks_per_iter_for_test


    def setup(self, stage):
        # assign datapts to train/val/test split
        # todo: split class-labels in to train/val/test class-labels
        # self.train_classes = train_classes
        # self.val_classes = val_classes
        # self.test_classes = test_classes

    #     self.meta_train_dset =
    #     self.meta_val_dset =
    #     self.meta_test_dset =
        pass

    def train_dataloader(self):
        train_dl = KShotDataLoader(
            dataset=self.meta_train_dset,
            batch_size=self.num_tasks_per_iter_for_train,
            max_iter=self.max_iter_for_train,
            **self.shared_dl_config
        )
        return train_dl

    def val_dataloader(self):
        val_dl = KShotDataLoader(
            dataset=self.meta_val_dset,
            batch_size=self.num_tasks_per_iter_for_val,  # todo: [x] check if other more reasonable choice? --> found an ans to this question --> see roam 2022-01-13 (r)
            max_iter=self.max_iter_for_val,
            **self.shared_dl_config
        )
        return val_dl

    def test_dataloader(self):
        test_dl = None
        if self.meta_test_dset is not None:
            test_dl = KShotDataLoader(
                dataset=self.meta_test_dset,
                batch_size=self.num_tasks_per_iter_for_test,
                max_iter=self.max_iter_for_test,
                **self.shared_dl_config
            )
        return test_dl


# helper to create an ImageDataset object given classes indices
def create_dset_from_classes(imgs: np.ndarray,
                             targets: List[int],
                             selected_classes: List[int],
                             img_xform: Optional[Callable] = None) -> Dataset:
    """Given an array of imgs, create a dataset that is of one of the given classes
    Args
    ----
    imgs : np.ndarray
    targets : List[int]
    selected_classes : iterable[int]
        List of class indices to constructe the dataset; i.e, the returned Dataset
        contains all images of these targets, and no images of any other classes

    Returns
    -------
        (ImageDataset) containing a subset of `imgs` that are of one of the classes
        in `targets`
    """
    is_member = [c in selected_classes for c in targets]
    return KShotImageDataset(imgs=imgs[is_member],
                             targets=targets[is_member],
                             img_xform=img_xform)


