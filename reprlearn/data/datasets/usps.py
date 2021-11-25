from typing import Any, Dict, Tuple
from torchvision.datasets import USPS


class USPSDataset(USPS):

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
        return {'x': img, 'y': target}

    @classmethod
    def unpack(cls, batch: Dict[str,Any]) -> Tuple[Any,Any]:
        # todo: consider if this is a good design
        # - if batch is a single sample from MNIST Dataset (rather than a batch from
        # MNISTDataModule's any dataloaders, e.g., ) the output of this method
        # is a single image, and a single label (ie. no batch dimension)
        return batch['x'], batch['y']