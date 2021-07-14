import joblib
from pathlib import Path
from typing import Any,Tuple, Optional,  Union, Callable, Dict, Iterable, List
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
from torchvision import transforms
import torch
from .two_factor_dataset import TwoFactorDataset
from .rotated_mnist import RotatedMNIST

class MultiRotatedMNIST(TwoFactorDataset):

    _name_formatspec = "{mode}-MNIST-{angles_str}"
    _keys = RotatedMNIST._keys

    def __init__(self,
            data_root: Union[Path, str],
            angles: List[float],
            selected_inds: Iterable[int] = None,
            transform: Optional[Callable] = None,
            digit_label_transform: Optional[Callable] = None,
            angle_label_transform: Optional[Callable] = None,
            train: bool=True,
            download: bool = True,
    ):
        """
        :param data_root:  root dir that contains MNIST folder
        :param angles:
        :param transform:
        :param digit_label_transform:
        :param angle_label_transform:
        :param train:
        :param download:
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.angles = sorted(angles)
        self.angle2idx = {angle:i for i, angle in enumerate(self.angles)}
        self.idx2angle = {i:angle for i, angle in enumerate(self.angles)}

        # Optionally, set the indices of original MNIST dataset to be selected in this Dataset
        self.selected_inds = selected_inds

        # Extra transform that will be applied after the base transforms, ToTensor() and TF.rotate
        self.transform = transform
        self.digit_label_transform = digit_label_transform
        # By default, encode the angle(float) to an integer angle-label (ie. style class index)
        self.angle_label_transform = angle_label_transform or \
                                     transforms.Lambda(lambda angle_label: self.angle2idx[float(angle_label)]) #float conversion because DataLoader's collate_fn will make it return torch.tensor()
        self.train = train
        self.mode = 'train' if self.train else 'test'
        self.download = download

        # Create a concatenated dataset from multiple MonoMNIST datasets as
        # specified in the input argument `colors`
        self.dsets = self.get_rotated_dsets(self.angles)
        self.ds = ConcatDataset(self.dsets)

    def get_rotated_dsets(self, angles: List[float]) -> List[RotatedMNIST]:
        dsets = []
        for angle in angles:
            ds = RotatedMNIST(
                data_root=self.data_root,
                angle=angle,
                selected_inds=self.selected_inds,
                transform=self.transform,
                digit_label_transform=self.digit_label_transform,
                angle_label_transform=self.angle_label_transform,
                train=self.train,
                download=self.download,
            )
            dsets.append(ds)
        return dsets

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index:int) -> Dict[str, Union[torch.Tensor, int, float]]:
        return self.ds[index]

    @property
    def name(self) -> str:
        return self._name_formatspec.format(
            mode=self.mode,
            angles_str='-'.join([str(angle) for angle in self.angles]),
        )


