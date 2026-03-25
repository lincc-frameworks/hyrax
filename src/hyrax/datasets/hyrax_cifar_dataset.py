# ruff: noqa: D101, D102
import logging
from pathlib import Path

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
        image, _ = self.cifar[idx]
        return image.numpy()

    def get_label(self, idx):
        """Get the label at the given index."""
        _, label = self.cifar[idx]
        return label

    def get_object_id(self, idx):
        """Get the object ID for the item as a string."""
        return f"{idx:0{self.id_width}d}"

    def __len__(self):
        return len(self.cifar)


    @staticmethod
    def collate(batch):
        """Collate a batch of samples into a single batch.

        This method takes a list of samples and collates them into a single batch.
        The returned batch will contain the following keys:

        - ``data``: A dictionary containing the collated data for the batch.
            - ``object_id``: A list of object IDs for the samples in the batch.
            - ``image``: A list of images for the samples in the batch.
            - ``label``: A list of labels for the samples in the batch (if provided).

        Parameters
        ----------
        batch : list
            A list of samples to collate.

        Returns
        -------
        dict
            A dictionary containing the collated batch data and metadata.

        """
        collated_data = {

            "object_id": [sample["data"]["object_id"] for sample in batch],
            "image": [sample["data"]["image"] for sample in batch],
        }

        if "label" in batch[0]["data"]:
            collated_data["label"] = [sample["data"]["label"] for sample in batch]

        return {
            "data": collated_data
        }
