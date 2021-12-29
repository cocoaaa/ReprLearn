from collections import defaultdict
from typing import Iterable, Optional, Callable, List, Dict, Tuple
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit

class ImageDataset(Dataset):

    def __init__(self,
                 imgs: np.ndarray,
                 classes: List[int],
                 img_xform: Optional[Callable] = None):
        """
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
        super().__init__()
        self.imgs = imgs
        self.targets = classes
        self.img_xform = img_xform

    def __getitem__(self, ind: int):
        img = self.imgs[ind]
        target = self.targets[ind]
        if self.img_xform is not None:
            img = self.img_xform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

    def create_task_set(
            self,
            task_classes: List[int],
            query_ratio: float=0.2,
            seed=123,
        )-> Dict[str, Dataset]:
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

# helper to create an ImageDataset object given classes indices
def create_dset_of_classes(imgs: np.ndarray,
                           classes: Iterable[int],
                           selected_classes: List[int],
                           img_xform: Optional[Callable] = None) -> Dataset:
    """Given an array of imgs, create a dataset that is of one of the given classes
    Args
    ----
    imgs : np.ndarray
    classes : List[int]
    selected_classes : iterable[int]
        List of class indices to constructe the dataset; i.e, the returned Dataset
        contains all images of these classes, and no images of any other classes

    Returns
    -------
        (ImageDataset) containing a subset of `imgs` that are of one of the classes
        in `classes`
    """
    is_member = [c in selected_classes for c in classes]
    return ImageDataset(imgs=imgs[is_member],
                        classes=classes[is_member],
                        img_xform=img_xform)



