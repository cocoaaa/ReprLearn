import joblib
from pathlib import Path
from typing import Any,Tuple, Optional,  Union, Callable, Dict, Iterable, List
from collections import defaultdict
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from reprlearn.data.transforms.transforms import  Monochromizer
from .two_factor_dataset import TwoFactorDataset

class MonoMNIST(TwoFactorDataset):
    """
    - data_root: root dir that contains "mnist_{color}.pkl" files
    - color: str; one of "red", "green", "blue"
    - transform: a callable that works on a torch tensor for 3-channel image (3,32,32)
    - target_transform: a callable that works on a int input for digit id (0, ..., 9)
    - colorstr_transform: a callable that works on a string type input (eg. "red", "blue")
    - download
    - seed
    - use_train_dataset

    Example
    -------
    - data_root: Path('/data/hayley-old/Tenanbaum2000/data/Mono-MNIST/')
        - mnist_data_root: data_root.parent
    """
    _fn_formatspec = "{mode}_mnist_{color}_seed-{seed}.pkl"
    _base_xform = transforms.ToTensor()
    _keys = ["img", "digit", "color"] # keys of the item(a dictionary) returned by __getitem__

    def __init__(
            self,
            data_root: Union[Path, str],
            color: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            colorstr_transform: Optional[Callable] = None,
            download: bool = True,
            seed: int=123,
            train: bool=True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.color = color.lower()
        # self._check_color_name() #TODO:
        assert self.color in ["gray", "red", "green", "blue"], "color must be one of gray, red, green, blue"

        # If transform is given, append it to the base transform
        self.transform = transforms.Compose([self._base_xform,
                                             Monochromizer(self.color)])
        if transform is not None:
            self.transform = transforms.Compose([self.transform, transform])

        self.target_transform = target_transform
        self.colorstr_transform = colorstr_transform
        self.seed = seed
        self.train = train
        self.mode = 'train' if self.train else 'test'

        # Load four split MNIST subsets as a dictionary of
        # k=digit_id (int) and v=List[PIL.Image]
        bn = self.make_basename()
        fn = self.data_root / bn
        if not fn.exists() and download:
            print(f"{fn} doesn't exist)"
                  f"\nStart processing to split the MNIST dataset into 4 subsets...")
            mnist_data_root = data_root.parent
            self.save_split_mnist(mnist_data_root, data_root, seed=self.seed,
                                  use_train_dataset=train)

        self.data, self.targets = joblib.load(fn)

    def __getitem__(self, index: int) -> Dict[str,Any]:
        """
        Parameters
        ----------
            index (int): Index

        Returns
        -------
            Dict: {
                "img": tensor_image, # by default, range of [0,1], shape (28,28).
                                     # Additional transform specified at initiation will be applied on top of the default transforms
                "digit": target,
                "color" : str_dataset_name/color
            } where target is index of the target class.
        """
        img, digit = self.data[index], int(self.targets[index])
        # img is a PIL Image of mode ('L'); unit8 in [0,...,255]; shape (28,28)

        if self.transform is not None:  # always perform the required basic transforms (ie. to Tensor and monochromizer)
            img = self.transform(img)

        if self.target_transform is not None:
            digit = self.target_transform(digit)

        color = self.color
        if self.colorstr_transform is not None:
            color = self.colorstr_transform(color)

        sample = {
            "img": img,
            "digit": digit,
            "color": color
        }

        return sample

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def keys(cls) -> List[str]:
        return cls._keys

    @property
    def name(self) -> str:
        return self.make_basename()[:-4] # Drop suffix

    def make_basename(self):
        return self._fn_formatspec.format(mode=self.mode,
                                          color=self.color,
                                          seed=self.seed)

    @classmethod
    def save_split_mnist(cls,
                         mnist_data_root: Union[str,Path],
                         out_dir: Union[str,Path]=None,
                         seed: Optional[int]=None,
                         use_train_dataset:bool=True):
        """
        Split the original MNIST dataset into 4 non-overlapping subsets of (almost)
        equal size, using a random-split.

        Note
        ----
            This method only partitions the full MNIST into 4 non-overlapping partitions, where each partition has the same
        distribution of the content-ids (ie. digit-ids, ie. [n0, n1, ..., n9] is (almost) the same across all partitions.
            The purpose of each partition is to be colorized into different color (r,g,b,gray)

        TODO:
        :param mnist_data_root: A directory that contains MNIST/raw and MNIST/processed.
            Eg. '/data/hayley-old/Tenanbaum2000/data/'
        :param out_dir: A directory to save the dictionries as pickle files.
            Eg. Path('/data/hayley-old/Tenanbaum2000/data/Mono-MNIST/')
        :param seed: Use an int to set a manual seed for a random generator that splits the
            origianl MNIST dataset into 4 subsets. Default: None
        :param use_train_dataset: True to use MNIST's training dataset, else use test dataset
        :return: None; Saves the 4 subset datasets as a dictionary of
            key=digit_id (int) and value=List[PIL.Image]
        """
        out_dir = Path(out_dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True)
        mode = 'train' if use_train_dataset else 'test'
        ds = datasets.MNIST(mnist_data_root, train=use_train_dataset, download=True)
        digits = defaultdict(list)
        for i in range(len(ds)):
            x, y = ds[i]  # PIL.Image of mode 'L' (ie. grayscale); y is int
            digits[y].append(x)
        print("Original MNIST: num images per digit")
        for k, v in digits.items():
            print(k, len(v))
        print("="*10)

        dict_gray = {}
        dict_r = {}
        dict_g = {}
        dict_b = {}
        for digit, imgs in digits.items():
            n = len(imgs)
            n_color = n // 4
            n_gray = n - 3 * n_color
            gray_imgs, r_imgs, g_imgs, b_imgs = random_split(imgs,
                                                           [n_gray, n_color, n_color, n_color],
                                                           generator=torch.Generator().manual_seed(seed))
            # gray/r/g/b: list of pil images -- each dataset has the same digit id; different styles (either grayscale, red, green, blue)

            print("Digit id: ", digit)
            print("\t", [len(i) for i in [gray_imgs, r_imgs, g_imgs, b_imgs]])

            dict_gray[digit] = gray_imgs
            dict_r[digit] = r_imgs
            dict_g[digit] = g_imgs
            dict_b[digit] = b_imgs

        X_gray, Y_gray = cls.dict_imgs2tuple_xy(dict_gray)
        X_r, Y_r = cls.dict_imgs2tuple_xy(dict_r)
        X_g, Y_g = cls.dict_imgs2tuple_xy(dict_g)
        X_b, Y_b = cls.dict_imgs2tuple_xy(dict_b)

        def XY(color):
            return {
                "gray": (X_gray, Y_gray),
                "red": (X_r, Y_r),
                "green": (X_g, Y_g),
                "blue": (X_b, Y_b)
            }[color]

        for color in ['gray', 'red', 'green', 'blue']:
            bn = cls._fn_formatspec.format(mode=mode,
                                        color=color,
                                        seed=seed)
            out_fn = out_dir / bn

            joblib.dump(XY(color), out_fn)
            print("Saved: ", out_fn)

    @classmethod
    def unpack(cls, batch: Dict[str, Any]) -> Tuple[Any]:
        """Unpacks a batch as a dictionary to a tuple of (data_x, content_label, style_label),
        so that the dataloading implementation is similar to standard torch's dataset objects

        Parameters
        ----------
        batch : Dict[str,Any]
            a sample from the dataset returned by self.__getitem__(), containing the data("x"),
            content-label and style-label

        Returns
        -------
        a tuple of the data, its content label and its style label
        """
        return (batch["img"], batch["digit"], batch["color"])

    @staticmethod
    def dict_imgs2tuple_xy(dict_imgs: Dict[int, List],
                           verbose=False) -> Tuple[Iterable, Iterable]:
        pil_imgs = []
        ys = []
        for k, v in dict_imgs.items():
            n_imgs = len(v)
            pil_imgs.extend(v)
            ys.extend([k] * n_imgs)

            if verbose:
                print("\n", k)
                print(len(pil_imgs), len(ys))

        return (pil_imgs, ys)
