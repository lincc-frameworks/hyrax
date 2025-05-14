import numpy as np
import torch.nn as nn
from torch import from_numpy
from torch.utils.data import Dataset, IterableDataset

from hyrax.data_sets import HyraxDataset
from hyrax.models import hyrax_model


@hyrax_model
class LoopbackModel(nn.Module):
    """Simple model for testing which returns its own input"""

    def __init__(self, config, shape):
        super().__init__()
        # This is created so the optimizier can find at least one weight
        self.unused_module = nn.Conv2d(1, 1, kernel_size=1, stride=0, padding=0)
        self.config = config

    def forward(self, x):
        """We simply return our input"""
        return x

    def train_step(self, batch):
        """Training is a noop"""
        return {"loss": 0.0}


class RandomDataset(HyraxDataset, Dataset):
    """Dataset yielding pairs of random numbers. Requires a seed to emulate
    static data on the filesystem between instantiations"""

    def __init__(self, config):
        size = config["data_set"]["size"]
        seed = config["data_set"]["seed"]
        rng = np.random.default_rng(seed)
        self.data = rng.random((size, 2), np.float32)

        # Start our IDs at a random integer between 0 and 100
        id_start = rng.integers(100)
        self.id_list = list(range(id_start, id_start + size))

        super().__init__(config)

    def __getitem__(self, idx):
        return from_numpy(self.data[idx])

    def __len__(self):
        return len(self.data)

    def ids(self):
        """Yield IDs for the dataset"""
        for id_item in self.id_list:
            yield str(id_item)


class RandomIterableDataset(RandomDataset, IterableDataset):
    """Iterable version of RandomDataset"""

    def __iter__(self):
        for item in self.data:
            yield from_numpy(item)
