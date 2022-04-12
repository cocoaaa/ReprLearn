# Source: https://github.com/JStehouwer/FFD_CVPR2020/blob/af5163559433641d40431d6276bd21aab8530536/dataset.py#L75
# Title: On the Dection of Digial Face Manipulation
# CiteKey: Stehouwer2020OnTD
from imageio import imread
import os
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from typing import Tuple, Union, List, Iterable
# DATABASE = '/home/jstay/FFD/data/'
DATABASE = Path('/data/hayley-old/Github/Reverse_Engineering_GMs/data/for-exps/deepfakes/')
# <DATABASE>
# |- <datatype> train
#     |- Real
#     |- Fake
# |- <datatype> test
#     |- Real
#     |- Fake

DATASETS = {
    'Real': 0,
    'Fake': 1
}


class RealOrFakeDataset:

    def __init__(self, label_name:str , label:int, datatype: str,
                 img_size:Tuple[int,int], norm: Union[List, torch.Tensor],
                 seed:int, bs:int, drop_last:bool,
                 num_workers=2, pin_memory=True):
        """

        :param label_name: (str) name of the dataset;'Real' or 'Fake'
        :param label: (int) class label for the images; 0 for 'Real', 1 for 'Fake'
        :param datatype: (str) 'train' or 'eval' or 'test' # i think 'eval' is for validation
        :param img_size:
        :param norm:
        :param seed:
        :param bs:
        :param drop_last:
        """
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(*norm)
        ])

        self.label_name = label_name
        self.label = label
        self.datatype = datatype
        self.data_dir = DATABASE/self.datatype/self.label_name #e.g.: data/DTTD/train/Fake or data/DTTD/train/Real #so we have to name the input dataset folders this way
        files = os.listdir(self.data_dir)
        if self.datatype != 'test':
            random.Random(seed).shuffle(files)
        self.images = [self.data_dir/fn for fn in files]
        self.loader = DataLoader(self, num_workers=num_workers, batch_size=bs, shuffle=(self.datatype != 'test'),
                                 drop_last=drop_last, pin_memory=pin_memory)
        self.generator = self.get_batch()

        print(f'Constructed Dataset {self.data_dir} of size {len(self)}')

    def load_image(self, path):
        return self.transform(imread(path))

    def load_mask(self, path):
        return self.transform_mask(imread(path))

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.load_image(fn)
        return {'x': img, 'y': self.label}# , 'fn': fn}

    def __len__(self):
        return len(self.images)

    def get_batch(self):
        if self.datatype == 'test':
            for batch in self.loader:
                yield batch
        else:
            while True: # interesting; i guess this makes us to keep loop over the dataset more than one iteration of the iterator
                for batch in self.loader:
                    yield batch


class DFFDDataset:

    def __init__(self, datatype: str, bs: int, img_size: Tuple[int,int],
                 norm: Union[torch.Tensor, Iterable], seed:int=1,
                 num_workers=2, pin_memory=True):
        """
        :param datatype: (str) 'train' or 'eval' or 'test'
        # i think 'eval' is for validation
        :param bs:
        :param img_size:
        :param norm:
        :param seed:
        """
        # self.batch_size = bs if datatype == 'test' else 2*bs #todo: only true when we call the `get_batch` of this class without index kwg specified
        self.batch_size = 2*bs # `bs` num of reals and `bs` num of fakes
        drop_last = datatype == 'train'

        # datasets contains two datasets (one for all reals, the other for all fakes
        # [DTTDDataset for Real images, DTTDDataset for fake images]
        datasets = [RealOrFakeDataset(real_or_fake, DATASETS[real_or_fake], datatype,
                                        img_size, norm,
                                        seed, bs, drop_last,
                                      num_workers=num_workers, pin_memory=pin_memory) \
                     for real_or_fake in DATASETS]
        drop_last = datatype == 'train' or datatype == 'eval' # 'eval' is for validation?
        self.datasets = datasets

    def __len__(self):
        return min(list(map(len, self.datasets)))

    def get_batch(self, index=-1):
        batch = None
        if index == -1: #self.datatype != 'test':
            batch = [next(real_or_fake_dset.generator, None) \
                     for real_or_fake_dset in self.datasets]
        else: # todo: get rid of this part because we want to evaluate on both fake detection and real detection rates (?)
            batch = [next(self.datasets[index].generator, None)]
        if any([_ is None for _ in batch]):
            return None
        batch_x = torch.cat([_['x'] for _ in batch], dim=0).cuda()
        batch_y = torch.cat([_['y'] for _ in batch], dim=0).cuda()
        # batch_fns = torch.cat([_['fn'] for _ in batch], dim=0)
        return {'x': batch_x, 'y': batch_y}
