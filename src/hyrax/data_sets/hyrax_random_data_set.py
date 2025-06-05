import numpy as np
from astropy.table import Table
from torch import from_numpy
from torch.utils.data import Dataset, IterableDataset

from .data_set_registry import HyraxDataset

#! Probably want to give this the same treatment as HyraxCifarDataset, and create
#! a HyraxRandomBase class, and then two subclasses for map-like and iterable datasets.


class HyraxRandomDataset(HyraxDataset, Dataset):
    """Dataset yielding pairs of random numbers. Requires a seed to emulate
    static data on the filesystem between instantiations"""

    def __init__(self, config):
        # The total number of random data samples produced
        data_size = config["data_set"]["size"]
        if not isinstance(data_size, int):
            raise ValueError(f"Expected integer value for `config['data_set']['size']`, but got {data_size}")

        # The shape of each random data sample as a tuple.
        # i.e. (3, 29, 29) = 3 layers of 2d data, each layer is 29x29 elements.
        data_shape = tuple(config["data_set"]["shape"])
        if not len(data_shape):
            raise ValueError("Expected `config['data_set']['data_shape']` to have at least 1 value.")

        for e in data_shape:
            if e < 1:
                raise ValueError(
                    f"Expected all values in `config['data_set']['data_shape']`\
                        to be > 0, but got {data_shape}."
                )
            if not isinstance(e, int):
                raise ValueError(
                    f"Expected all values in `config['data_set']['data_shape']`\
                        to be integers, but got {data_shape}."
                )

        # Random seed to use for reproducibility
        seed = config["data_set"]["seed"]
        rng = np.random.default_rng(seed)

        # Note: We raise exceptions if data_size is not an int, so we can assume
        # that turning that into a tuple and adding `data_shape` should work.
        self.data = rng.random((data_size,) + data_shape, np.float32)

        # Start our IDs at a random integer between 0 and 100
        id_start = rng.integers(100)
        self.id_list = list(range(id_start, id_start + data_size))

        # Create a metadata_table that is used when visualizing data
        metadata_table = Table({"object_id": np.array(list(self.ids()))})

        super().__init__(config, metadata_table)

    def __getitem__(self, idx):
        """Return a torch.Tensor object at the index"""
        return from_numpy(self.data[idx])

    def __len__(self):
        """Get the total number of samples in this dataset"""
        return len(self.data)

    def ids(self):
        """Yield IDs for the dataset"""
        for id_item in self.id_list:
            yield str(id_item)


class HyraxRandomIterableDataset(HyraxRandomDataset, IterableDataset):
    """Iterable version of RandomDataset"""

    def __iter__(self):
        for item in self.data:
            yield from_numpy(item)
