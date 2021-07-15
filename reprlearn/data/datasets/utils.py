from PIL.Image import Image
from typing import Tuple, List, Union
from pathlib import Path
from torchvision import datasets


def get_mnist_data(
        data_root: Union[Path,str],
        use_train_dataset: bool = True,
        download=True
) -> Tuple[List[Image], List[int]]:
    # Get all imgs and digits in MNIST dataset
    ds = datasets.MNIST(data_root, train=use_train_dataset, download=download)

    pil_imgs = []
    digit_labels = []
    for i in range(len(ds)):
        x, y = ds[i]
        pil_imgs.append(x)
        digit_labels.append(y)
    return pil_imgs, digit_labels

