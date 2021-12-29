from collections import Counter
from pprint import pprint
import numpy as np
from reprlearn.data.datasets.base import ImageDataset
from reprlearn.data.samplers.kshot_sampler_v0 import KShotSampler


def create_dummy_imagedataset(n_dpts=100, n_classes=10) -> ImageDataset:
    h, w, nc = 32, 32, 3
    imgs = np.zeros((n_dpts, h, w, nc))
    # dummy targets as random class from 10 class labels
    classes = np.random.randint(low=0, high=n_classes, size=n_dpts, dtype=int)

    return ImageDataset(imgs, classes)


def test_create_task_set_3ways():
    dset = create_dummy_imagedataset()
    task_classes = [0, 2, 4]
    task_set = dset.create_task_set(task_classes)
    support, query = task_set['support'], task_set['query']

    # check if the support set is balanced
    # ie. each class has a similar number of images
    print('=== Num. images per class (Support set) === ')
    pprint(Counter(support.targets))
    print('=== Num. images per class (Query set) ===')
    pprint(Counter(query.targets))


def test_create_task_set_5ways():
    dset = create_dummy_imagedataset()
    task_classes = [0, 2, 4, 6, 8]
    task_set = dset.create_task_set(task_classes)
    support, query = task_set['support'], task_set['query']

    # check if the support set is balanced
    # ie. each class has a similar number of images
    print('=== Num. images per class (Support set) === ')
    pprint(Counter(support.targets))
    print('=== Num. images per class (Query set) ===')
    pprint(Counter(query.targets))

def test_kshot_sampler(n_imgs_per_class = 5):
    dset = create_dummy_imagedataset()
    sampler = KShotSampler(k_shot = n_imgs_per_class)
    sample = sampler.sample(dset)
    #todo
    print("classes in the sample: ")
    print([dpt[1] for dpt in sample])


def test_check_done_false_1():
    n_imgs_per_class = 5
    sampler = KShotSampler(k_shot = n_imgs_per_class)
    inds_per_class = {0: [1,2,3],
                      1: [4,5],
                      2: [6]}
    assert not sampler.check_done(inds_per_class)


def test_check_done_false_2():
    n_imgs_per_class = 3
    sampler = KShotSampler(k_shot = n_imgs_per_class)
    inds_per_class = {0: [1,2,3],
                      1: [4,5],
                      2: [6]}
    assert not sampler.check_done(inds_per_class)


def test_check_done_false_3():
    n_imgs_per_class = 3
    sampler = KShotSampler(k_shot=n_imgs_per_class)
    inds_per_class = {0: [1, 2, 3],
                      1: [4, 5, 6],
                      2: [7]}
    assert not sampler.check_done(inds_per_class)


def test_check_done_true_1():
    n_imgs_per_class = 3
    sampler = KShotSampler(k_shot=n_imgs_per_class)
    inds_per_class = {0: [1, 2, 3],
                      1: [4, 5, 6],
                      2: [7, 8, 9]}
    assert sampler.check_done(inds_per_class)


def test_kshot_sampler_get_support_and_query(n_imgs_per_class = 5):
    dset = create_dummy_imagedataset(500)
    sampler = KShotSampler(k_shot = n_imgs_per_class)
    task_set = sampler.get_support_and_query(dset)
    support, query = task_set['support'], task_set['query']
    print("=== Support sample ===")
    print("\tLabel counts: ", Counter([dpt[1] for dpt in support]))
    print("=== Query sample ===")
    print("\tLabel counts: ", Counter([dpt[1] for dpt in query]))


if __name__ == '__main__':
    # test_imagedataset()
    # test_create_task_set_3ways()
    # test_create_task_set_5ways()
    # test_kshot_sampler()
    # test_check_done_false_1()
    # test_check_done_false_2()
    # test_check_done_false_3()
    # test_check_done_true_1()
    test_kshot_sampler_get_support_and_query()