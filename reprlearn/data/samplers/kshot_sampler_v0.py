from reprlearn.data.datasets.base import ImageDataset
from collections import defaultdict
from typing import Iterable, Optional, Callable, List, Dict, Tuple
import numpy as np

# ===============
# K-shot sampler
# ===============
# def k_shot_sampler(dset: ImageDataset,
#                    k_shot: int,
#                    shuffle=True):
#     """Given the dataset of labelled images, return the indices for sampling
#     `k-shot` number of images per class in the dataset's classes.
#     If shuffle, we shuffle the indices of the dset for each call to the iterator
#
#     Args
#     ----
#     dset : ImageDataset
#     k : int
#         number of images per class
#     shuffle : bool (default True)
#
#     Returns:
#     (List[int]) : indices to the datapts to sample for this iteration
#     """
#     def check_done(inds_per_class):
#         for ind_list in inds_per_class.values():
#             if len(ind_list) < k_shot:
#                 return False
#         return True
#
#     targets = dset.targets
#     inds = list(range(len(dset)))
#     if shuffle:
#         np.random.shuffle(inds)  # shuffle in-place
#
#     inds_per_class = defaultdict(list)
#     for i in inds:
#         c = targets[i]
#         if len(inds_per_class[c]) < k_shot:
#             inds_per_class.append(i)
#
#         if check_done(inds_per_class):
#             break
#     sample_inds = np.array(inds_per_class.values).flatten()
#     np.random.shuffle(sample_inds)  # we don't want to load imgs for one-class all in a row, and then next class's images in a row, etc
#     return sample_inds

class KShotSampler():

    def __init__(self, k_shot: int, shuffle:bool=True) -> None:
        """Given the dataset of labelled images, return the indices for sampling
            `k-shot` number of images per class in the dataset's classes.
            If shuffle, we shuffle the indices of the dset for each call to the iterator

            Args
            ----
            dset : ImageDataset
            k : int
                number of images per class
            shuffle : bool (default True)
        """
        self.k_shot = k_shot
        self.shuffle = shuffle

    def check_done(self,
                   inds_per_class: Dict[int, List],
                   num_per_class: Optional[int] = None,
                   ):
        num_per_class = num_per_class or self.k_shot
        for ind_list in inds_per_class.values():
            if len(ind_list) != num_per_class:
                return False
        return True

    def sample(
            self,
            dset: ImageDataset,
            num_per_class: Optional[int]=None,
            collate_fn: Optional[Callable]=None
    ) -> List[Tuple]: # [(x,y),...]  #List[int]:
        """Given the dataset of labelled images, return the indices for sampling
        `k-shot` number of images per class in the dataset's classes.
        If shuffle, we shuffle the indices of the dset for each call to the iterator

        Returns:
        (List[int]) : indices to the datapts to sample for this iteration
        """
        num_per_class = num_per_class or self.k_shot
        n_ways = len(np.unique(dset.targets))
        if num_per_class * n_ways > len(dset.targets):
            raise ValueError

        inds = list(range(len(dset)))
        if self.shuffle:
            np.random.shuffle(inds)  # shuffle in-place

        inds_per_class = defaultdict(list)
        done_for_class = defaultdict(lambda: False)
        for i in inds:
            c = dset.targets[i]
            if not done_for_class[c]: #len(inds_per_class[c]) < num_per_class:
                inds_per_class[c].append(i)
                if len(inds_per_class[c]) == num_per_class:
                    done_for_class[c] = True #done collecting dpts for this class
                    # print("Done for class: ", c)

            # if self.check_done(inds_per_class, num_per_class):
            #     break
            if np.alltrue(np.fromiter(done_for_class.values(), dtype=bool)):
                break

        print("Done collecting datapts for each class...")
        for k,v in inds_per_class.items():
            # print(k, len(v))
            if len(v) != num_per_class:
                raise ValueError

        sample_inds = np.stack(
            [np.fromiter(ilist, dtype=int) for ilist in inds_per_class.values()]
        ).flatten()
        # print("sample_inds: ", sample_inds)
        np.random.shuffle(sample_inds)  # we don't want to load imgs for one-class all in a row, and then next class's images in a row, etc

        sample = [dset[i] for i in sample_inds] # apply current dataset's image transform if specified
        if collate_fn is not None:
            sample = collate_fn(sample)
        return sample

    def get_support_and_query(
        self,
        dset: ImageDataset,
        collate_fn: Optional[Callable] = None
    ) -> Dict:
        sample = self.sample(dset,
                             num_per_class=2*self.k_shot,
                             collate_fn=collate_fn)
        n_way = len(np.unique(dset.targets))
        n_support = self.k_shot * n_way
        return {'support': sample[:n_support],
                'query': sample[n_support:]}


