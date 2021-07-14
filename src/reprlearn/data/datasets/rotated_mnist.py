from typing import Optional, Callable, Dict, Union, Iterable
from pathlib import Path
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from .utils import get_mnist_data
from .two_factor_dataset import TwoFactorDataset

class RotatedMNIST(TwoFactorDataset):

    _fn_formatspec = "{mode}_mnist_rot{angle}"
    _keys = ['img', 'digit', 'angle']

    def __init__(
            self,
            data_root: Path,
            angle: float,  # counter-clockwise,
            selected_inds: Iterable[int] = None,
            transform: Optional[Callable] = None,
            digit_label_transform: Optional[Callable] = None,
            angle_label_transform: Optional[Callable] = None,
            train: bool = True,
            download: bool = True,

    ):
        """Sets self.data and self.digit_labels attribute
        - self.data : List[PIL.Image]
        - self.digit_labels: List
        """
        print("angle: ", angle, " , type: ", type(angle))
        super().__init__()
        self.data_root = Path(data_root)
        self.angle = angle
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda timg: TF.rotate(timg, self.angle))
        ])
        # If transform is given, append it to the base transform above
        if transform is not None:
            self.transform = transforms.Compose([self.transform, transform])
        self.digit_label_transform = digit_label_transform  # input is int
        self.angle_label_transform = angle_label_transform  # input is float

        self.train = train
        self.mode = 'train' if self.train else 'test'
        self.download = download

        self.data, self.digit_labels = get_mnist_data(self.data_root, self.train, self.download)

        if selected_inds is not None:
            self.data = [self.data[i] for i in selected_inds]
            self.digit_labels = [self.digit_labels[i] for i in selected_inds]

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, int, float]]:
        """Returns a sample of type Dict with keys:
        sample['img'] : torch.Tensor of a single image
        sample['digit'] : int in [0, ..., 9]
        sample['angle'] : float
            self.angle for any item
        """
        img, digit_label = self.data[index], int(self.digit_labels[index])
        angle_label = self.angle
        # img is a PIL image of mode L, shape (28,28)
        if self.transform is not None:
            img = self.transform(img)
        if self.digit_label_transform is not None:
            digit_label = self.digit_label_transform(digit_label)
        if self.angle_label_transform is not None:
            angle_label = self.angle_label_transform(angle_label)  # may be used for encoding domain label when used with other angles dataset
        return {"img": img,
                "digit": digit_label,
                "angle": angle_label}

    def __len__(self) -> int:
        return len(self.data)

    @property
    def name(self) -> str:
        return self._fn_formatspec.format(mode=self.mode, angle=self.angle)

