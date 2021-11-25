# test how python override works
from typing import Iterable, Tuple, Dict, Any
from torchvision.datasets import MNIST, USPS

# super(`type`) delegates method calls to a parent or sibling *class* of
# `type`
class MyMNIST(MNIST):
    """Override __getitem__ to return a dictionary rather than the tuple
    """
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Args:
            index (int): Index

        Returns:
            dict: {'x': image} for unsupervised dataset
                or {'x': image, 'y': label} for classification type supervised dataset,
                or {'x': image, 'y': content_label, 'd': style/domain_label for dataset from multi-sources/styles/domains
        """
        img, target = super().__getitem__(index)
        return {'x': img,
                'y': target}

class MyUSPS(USPS):

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Args:
            index (int): Index

        Returns:
            dict: {'x': image} for unsupervised dataset
                or {'x': image, 'y': label} for classification type supervised dataset,
                or {'x': image, 'y': content_label, 'd': style/domain_label for dataset from multi-sources/styles/domains
        """
        img, target = super().__getitem__(index)
        return {'x': img,
                'y': target}


