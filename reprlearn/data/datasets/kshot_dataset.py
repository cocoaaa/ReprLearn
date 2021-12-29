from collections import defaultdict
from typing import Iterable, Optional, Callable, List, Dict, Tuple
import numpy as np
from .base import ImageDataset
from sklearn.model_selection import StratifiedShuffleSplit

class KShotImageDataset(ImageDataset):
    """This is a child class of ImageDataset.
    It additionally implements a method to create task set containing the
    support and query datasets from the given imgs and targets (class labels)

    Args
    ----
    imgs : np.ndarray
        Array of uint8 images; (N, 32, 32, 3)
    classes : List[int]
        List of integers indicating classes
    img_xform : Optional[Callable]
        Transformation to be applied to each item. First `transform` object
        applied should work with input of type `np.ndarray` (h,w,nc)
    """
    def create_task_set(
            self,
            task_classes: List[int],
            query_ratio: float=0.2,
            seed=123,
        )-> Dict[str, ImageDataset]:
        """Create a task set (which has two types of dataset for the same task
        (e.g., same classification task of N-way, K-shot for the same task-classes)
        which has support and query sets.
        The support set is used for 'adaptation': optimize the current, given
         meta-parameter to fit to this particular task.
        The query set is used to update the meta-parameter (so that the meta-programmer
        generates a good task-learner, where 'good' is measured by the generated
        task-learner's performance on this query set (equivalent to the usual 'test' set)

        Args
        ----
        query_ratio : float in [0,1]
            the ratio of the size of query_set to the size of the support_set
        seed : int
            random seed for split to create support, query sets

        Returns
        -------
        (Dict[str, ImageDataset]) : returns support and query datasets
        """
        is_p = 0 <= query_ratio and query_ratio <= 1
        assert is_p, "query_ratio must be a probability, ie. a float in range[0,1]"

        # filter the (x,y)'s so that y is a member of task_classes
        is_member = [c in task_classes for c in self.targets]
        task_imgs = self.imgs[is_member]
        task_ys = self.targets[is_member]

        # split to support, query images as the usual split of train/test dataset
        # for training a task-learner (ie. a classifier)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=query_ratio, random_state=seed)
        sss.get_n_splits(task_imgs, task_ys)
        for support_inds,  query_inds in sss.split(task_imgs, task_ys):
            support = ImageDataset(
                imgs=task_imgs[support_inds],
                classes=task_ys[support_inds],
                img_xform=self.img_xform
            )
            query = ImageDataset(
                imgs=task_imgs[query_inds],
                classes=task_ys[query_inds],
                img_xform=self.img_xform
            )
            return {'support': support, 'query': query}

