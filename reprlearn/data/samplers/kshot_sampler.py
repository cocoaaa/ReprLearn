from reprlearn.data.datasets.base import ImageDataset
from collections import defaultdict
from typing import Iterable, Optional, Callable, List, Dict, Tuple
import numpy as np

# ===============
# Returns a list of datapoints from the dataset so that
# the list contains the same number of datapoints per class (if possible)
# ===============
class KShotSampler():

    def __init__(self, shuffle:bool=True) -> None:
        """Given the dataset of labelled images, return the indices for sampling
            the same number of datapts per class for each class the dataset's targets.
            If shuffle, we shuffle the indices of the dataset before collecting
            the datapoints.

            Args
            ----
            dset : ImageDataset
            k_shot: int
                number of images per class
            shuffle : bool (default True)
        """
        pass

    def get_sample_inds_per_class(self,
                                  dset: ImageDataset,
                                  num_per_class: int,
                                  shuffle: bool=True,
                                  verify: bool=True,
                                ):
        """Given the dataset of labelled images, return the indices for sampling
       `num_per_class` number of images per class in the dataset's classes.
       If shuffle, we shuffle the indices of the dset for each call to the iterator

       Returns:
       (List[int]) : indices to the datapts to sample for this iteration
       """
        unique_classes = np.unique(dset.targets)
        n_ways = len(unique_classes)
        if num_per_class * n_ways > len(dset.targets):
            raise ValueError

        inds = list(range(len(dset)))
        if shuffle:
            np.random.shuffle(inds)  # shuffle in-place

        inds_per_class = {c:[] for c in unique_classes}
        done_for_class = {c:False for c in unique_classes}
        for i in inds:
            c = dset.targets[i]
            if not done_for_class[c]:  # len(inds_per_class[c]) < num_per_class:
                inds_per_class[c].append(i)
                if len(inds_per_class[c]) == num_per_class:
                    done_for_class[c] = True  # done collecting dpts for this class

            if np.alltrue(np.fromiter(done_for_class.values(), dtype=bool)):
                break

        print("Done collecting datapts for each class...")
        if verify:
            for c in np.unique(dset.targets):
                inds = inds_per_class[c]
                if len(inds) != num_per_class:
                    raise ValueError
        return inds_per_class

    def sample(self,
        dset: ImageDataset,
        num_per_class: int,
        shuffle: bool=True,
        collate_fn: Optional[Callable]=None
    ) -> List[Tuple]: # [(x,y),...]  #List[int]:
        """Given the dataset of labelled images, return the collection/list
        of datapoints from the dataset; the collection of datapoints (aka. sample)
        contains equal number of datapoints per class (with best effort)

        Args
        ----
        dset : ImageDataset
            source dataset to sample datapoints from
        num_per_class : int
            k in k-shot
        shuffle : bool
            if shuffle, shuffle the indices of the dataset before collecting
            the datapoints
        collate_fn : Callable
            Similar to the collating function in torch's DataLoader argument;
            It take a list of datapoints and apply it to turn the list into a
            desired form of 'batch'

        Returns:
        (Batch or List[datapts]) : A collection of datapts sampled
        """
        inds_per_class = self.get_sample_inds_per_class(dset, num_per_class, shuffle)

        sample_inds = np.stack(
            [np.fromiter(ilist, dtype=int) for ilist in inds_per_class.values()]
        ).flatten()
        # we don't want to load imgs for one-class all in a row,
        # and then next class's images in a row, etc
        np.random.shuffle(sample_inds)

        sample = [dset[i] for i in sample_inds] # apply current dataset's image transform if specified
        if collate_fn is not None:
            sample = collate_fn(sample)
        return sample

    def get_support_and_query(
            self,
            dset: ImageDataset,
            num_per_class: int,
            shuffle: bool=True,
            collate_fn: Optional[Callable] = None
    ) -> Dict:
        inds_per_class = self.get_sample_inds_per_class(dset, 2*num_per_class, shuffle)
        n_way = len(np.unique(dset.targets))
        support_inds = []
        query_inds = []
        for clabel, cinds in inds_per_class.items():
            cids = np.fromiter(cinds, dtype=int)
            support_inds.append(cids[:num_per_class])
            query_inds.append(cinds[num_per_class:])

        support_inds = np.array(support_inds)
        query_inds = np.array(query_inds)

        # we don't want to load imgs for one-class all in a row,
        # and then next class's images in a row, etc
        np.random.shuffle(support_inds)
        support_sample = [dset[i] for i in support_inds]  # apply current dataset's image transform if specified
        if collate_fn is not None:
            support_sample = collate_fn(support_sample)

        # Similarly for the query sample
        np.random.shuffle(query_inds)
        query_sample = [dset[i] for i in query_inds]  # apply current dataset's image transform if specified
        if collate_fn is not None:
            query_sample = collate_fn(query_sample)
        return {'support': support_sample,
                'query': query_sample}


