# ruff: noqa: D101, D102
import logging
from pathlib import Path

from PIL import Image

from .dataset_registry import HyraxDataset

logger = logging.getLogger(__name__)


class HyraxCifarDataset(HyraxDataset):
    """Map style CIFAR 10 dataset for Hyrax

    This utilizes the CIFAR dataset from torchvision for retrieving the dataset.
    """

    def __init__(self, config: dict, data_location: Path = None):
        import torchvision.transforms as transforms
        from torchvision.datasets import CIFAR10

        self.data_location = data_location

        self.training_data = config["data_set"]["HyraxCifarDataset"]["use_training_data"]

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        self.cifar = CIFAR10(
            root=self.data_location, train=self.training_data, download=True, transform=transform
        )

        n_id = len(self.cifar)
        self.id_width = len(str(n_id))

        super().__init__(config)

    def get_image(self, idx):
        """Get the image at the given index as a NumPy array."""
        img = self.cifar.data[idx]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.cifar.transform is not None:
            img = self.cifar.transform(img)

        return img.numpy()

    def get_label(self, idx):
        """Get the label at the given index."""

        label = self.cifar.targets[idx]
        if self.cifar.target_transform is not None:
            label = self.cifar.target_transform(label)

        return label

    def get_object_id(self, idx):
        """Get the object ID for the item as a string."""
        return f"{idx:0{self.id_width}d}"

    def __len__(self):
        return len(self.cifar)
