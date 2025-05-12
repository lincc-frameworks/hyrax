# ruff: noqa: D101, D102
import logging

import numpy as np
from torch.utils.data import Dataset, IterableDataset

from hyrax.config_utils import ConfigDict

from .data_set_registry import HyraxDataset

logger = logging.getLogger(__name__)


class HyraxCifarBase:
    """Base class for Hyrax Cifar datasets"""

    def __init__(self, config: ConfigDict):
        import torchvision.transforms as transforms
        from astropy.table import Table
        from torchvision.datasets import CIFAR10

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.cifar = CIFAR10(
            root=config["general"]["data_dir"], train=True, download=True, transform=transform
        )
        metadata_table = Table(
            {"label": np.array([self.cifar[index][1] for index in range(len(self.cifar))])}
        )
        super().__init__(config, metadata_table)


class HyraxCifarDataSet(HyraxCifarBase, HyraxDataset, Dataset):
    """Map style CIFAR 10 dataset for Hyrax

    This is simply a version of CIFAR10 that is initialized using Hyrax config with a transformation
    that works well for example code.

    We only use the training split in the data, because it is larger (50k images). Hyrax will then divide that
    into Train/test/Validate according to configuration.
    """

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        image, label = self.cifar[idx]
        return {
            "object_id": idx,
            "image": image,
            "label": label,
        }


class HyraxCifarIterableDataSet(HyraxCifarBase, HyraxDataset, IterableDataset):
    """Iterable style CIFAR 10 dataset for Hyrax

    This is simply a version of CIFAR10 that is initialized using Hyrax config with a transformation
    that works well for example code. This version only supports iteration, and not map-style access

    We only use the training split in the data, because it is larger (50k images). Hyrax will then divide that
    into Train/test/Validate according to configuration.
    """

    def __iter__(self):
        for idx, (image, label) in enumerate(self.cifar):
            yield {
                "object_id": idx,
                "image": image,
                "label": label,
            }
