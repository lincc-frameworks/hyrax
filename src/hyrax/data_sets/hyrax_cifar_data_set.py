# ruff: noqa: D101, D102
import logging
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, IterableDataset

from hyrax.config_utils import ConfigDict

from .data_set_registry import HyraxDataset

logger = logging.getLogger(__name__)


class HyraxCifarBase:
    """Base class for Hyrax Cifar datasets"""

    def __init__(self, config: ConfigDict, data_location: Path = None):
        import torchvision.transforms as transforms
        from astropy.table import Table
        from torchvision.datasets import CIFAR10

        self.data_location = data_location if data_location else config["general"]["data_dir"]

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.cifar = CIFAR10(root=self.data_location, train=True, download=True, transform=transform)
        metadata_table = Table(
            {"label": np.array([self.cifar[index][1] for index in range(len(self.cifar))])}
        )
        super().__init__(config, metadata_table)

    def get_image(self, idx):
        """Get the image at the given index as a NumPy array."""
        image, _ = self.cifar[idx]
        return image.numpy()

    def get_label(self, idx):
        """Get the label at the given index."""
        _, label = self.cifar[idx]
        return label

    def get_index(self, idx):
        """Get the index of the item."""
        return idx

    def get_object_id(self, idx):
        """Get the object ID for the item."""
        return idx


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
        return {
            "data": {
                "object_id": self.get_object_id(idx),
                "image": self.get_image(idx),
                "label": self.get_label(idx),
            },
            "object_id": self.get_object_id(idx),
        }


class HyraxCifarIterableDataSet(HyraxCifarBase, HyraxDataset, IterableDataset):
    """Iterable style CIFAR 10 dataset for Hyrax

    This is simply a version of CIFAR10 that is initialized using Hyrax config with a transformation
    that works well for example code. This version only supports iteration, and not map-style access

    We only use the training split in the data, because it is larger (50k images). Hyrax will then divide that
    into Train/test/Validate according to configuration.
    """

    def __iter__(self):
        for idx in range(len(self.cifar)):
            yield {
                "data": {
                    "object_id": self.get_object_id(idx),
                    "image": self.get_image(idx),
                    "label": self.get_label(idx),
                },
                "object_id": self.get_object_id(idx),
            }
