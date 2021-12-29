from collections import Counter
from pprint import pprint
from typing import Iterable, Dict, List, Optional
import numpy as np
from reprlearn.data.datasets.kshot_dataset import KShotImageDataset
from reprlearn.data.samplers.kshot_sampler import KShotSampler


def create_dummy_dataset(n_dpts=200, n_classes=10) -> KShotImageDataset:
    h, w, nc = 32, 32, 3
    imgs = np.zeros((n_dpts, h, w, nc))
    # dummy targets as random class from 10 class labels
    classes = np.random.randint(low=0, high=n_classes, size=n_dpts, dtype=int)
    return KShotImageDataset(imgs, classes)


def create_dummy_task_set(n_dpts=500,
                          n_classes=10,
                          task_classes: Optional[List[int]]=None
                          ) -> Dict[str, KShotImageDataset]:
    task_classes = task_classes or [0,2,4]
    dset = create_dummy_dataset(n_dpts, n_classes)
    task_set = dset.create_task_set(task_classes)
    return task_set


def test_create_task_set():
    task_classes = [0,2,4]
    task_set = create_dummy_task_set(task_classes=task_classes)
    support = task_set['support']
    query = task_set['query']

    for name, dset in task_set.items():
        print(f'==={name}===')
        print(dset.imgs.shape)
        print(len(dset.targets), np.unique(dset.targets))


def test_kshot_sampler(num_per_class, task_classes):
    task_set = create_dummy_task_set(task_classes=task_classes)
    support, query = task_set['support'], task_set['query']

    # Sampler to create a stratified sample set
    sampler = KShotSampler()
    support_sample = sampler.sample(
        dset=support,
        num_per_class=num_per_class
    ) # List of datapts

    print('Specified task classes: ', task_classes)
    print('Specified num_per_class: ', num_per_class)
    print('N imgs per class in support sample: ',
          Counter([dpt[1] for dpt in support_sample]))
    print([dpt[1] for dpt in support_sample])


def test_kshot_sampler_1shot_3way():
    num_per_class = 1
    task_classes = [0, 2, 4]
    task_set = create_dummy_task_set(task_classes=task_classes)
    support, query = task_set['support'], task_set['query']
    # Sampler to create a stratified sample set
    sampler = KShotSampler()
    support_sample = sampler.sample(
        dset=support,
        num_per_class=num_per_class
    ) # List of datapts

    assert set(np.unique(support.targets)) == set(task_classes)
    assert len(support_sample) == num_per_class * len(task_classes)

    print('Specified task classes: ', task_classes)
    print('Specified num_per_class: ', num_per_class)
    print('N imgs per class in support sample: ',
          Counter([dpt[1] for dpt in support_sample]))
    print([dpt[1] for dpt in support_sample])


def test_kshot_sampler_5shot_3way():
    num_per_class = 5
    task_classes = [0, 2, 4]
    task_set = create_dummy_task_set(task_classes=task_classes)
    support, query = task_set['support'], task_set['query']

    # Sampler to create a stratified sample set
    sampler = KShotSampler()
    support_sample = sampler.sample(
        dset=support,
        num_per_class=num_per_class
    ) # List of datapts

    print('Specified task classes: ', task_classes)
    print('Specified num_per_class: ', num_per_class)
    print('N imgs per class in support sample: ',
          Counter([dpt[1] for dpt in support_sample]))
    print([dpt[1] for dpt in support_sample])


def test_kshot_sampler_7shot_3way():
    num_per_class = 7
    task_classes = [0, 2, 4]
    task_set = create_dummy_task_set(task_classes=task_classes)
    support, query = task_set['support'], task_set['query']

    # Sampler to create a stratified sample set
    sampler = KShotSampler()
    support_sample = sampler.sample(
        dset=support,
        num_per_class=num_per_class
    ) # List of datapts

    print('Specified task classes: ', task_classes)
    print('Specified num_per_class: ', num_per_class)
    print('N imgs per class in support sample: ',
          Counter([dpt[1] for dpt in support_sample]))
    print([dpt[1] for dpt in support_sample])


def test_kshot_sampler_1shot_5way():
    num_per_class = 1
    task_classes = [0, 2, 4, 7, 9]
    task_set = create_dummy_task_set(task_classes=task_classes)
    support, query = task_set['support'], task_set['query']

    # Sampler to create a stratified sample set
    sampler = KShotSampler()
    support_sample = sampler.sample(
        dset=support,
        num_per_class=num_per_class
    )  # List of datapts

    print('Specified task classes: ', task_classes)
    print('Specified num_per_class: ', num_per_class)
    print('N imgs per class in support sample: ',
          Counter([dpt[1] for dpt in support_sample]))
    print([dpt[1] for dpt in support_sample])


def test_kshot_sampler_5shot_5way():
    num_per_class = 5
    task_classes = [0, 2, 4, 7, 9]
    task_set = create_dummy_task_set(task_classes=task_classes)
    support, query = task_set['support'], task_set['query']

    # Sampler to create a stratified sample set
    sampler = KShotSampler()
    support_sample = sampler.sample(
        dset=support,
        num_per_class=num_per_class
    )  # List of datapts

    print('Specified task classes: ', task_classes)
    print('Specified num_per_class: ', num_per_class)
    print('N imgs per class in support sample: ',
          Counter([dpt[1] for dpt in support_sample]))
    print([dpt[1] for dpt in support_sample])


def test_kshot_sampler_7shot_5way():
    num_per_class = 7
    task_classes = [0, 2, 4, 7, 9]
    task_set = create_dummy_task_set(task_classes=task_classes)
    support, query = task_set['support'], task_set['query']

    # Sampler to create a stratified sample set
    sampler = KShotSampler()
    support_sample = sampler.sample(
        dset=support,
        num_per_class=num_per_class
    )  # List of datapts

    print('Specified task classes: ', task_classes)
    print('Specified num_per_class: ', num_per_class)
    print('N imgs per class in support sample: ',
          Counter([dpt[1] for dpt in support_sample]))
    print([dpt[1] for dpt in support_sample])


# def test_kshot_sampler_get_support_and_query(n_imgs_per_class = 5):
#     dset = create_dummy_dataset(500)
#     sampler = KShotSampler(k_shot = n_imgs_per_class)
#     task_set = sampler.get_support_and_query(dset)
#     support, query = task_set['support'], task_set['query']
#     print("=== Support sample ===")
#     print("\tLabel counts: ", Counter([dpt[1] for dpt in support]))
#     print("=== Query sample ===")
#     print("\tLabel counts: ", Counter([dpt[1] for dpt in query]))


if __name__ == '__main__':
    # test_create_task_set()

    # 3ways
    test_kshot_sampler_1shot_3way()
    # test_kshot_sampler_5shot_3way()
    # test_kshot_sampler_7shot_3way()

    # 5ways
    # test_kshot_sampler_1shot_5way()
    # test_kshot_sampler_5shot_5way()
    # test_kshot_sampler_7shot_5way()

