import numpy as np
from astropy.table import Table
from torch import from_numpy
from torch.utils.data import Dataset, IterableDataset

from hyrax.data_sets.data_set_registry import HyraxDataset

INVALID_VALUES = {
    "nan": np.nan,
    "inf": np.inf,
    "-inf": -np.inf,
    "none": None,
}


class HyraxRandomDatasetBase:
    """Semi-private class that acts as the base class for `HyraxRandomDataset`
    and `HyraxRandomIterableDataset`."""

    def __init__(self, config):
        # The total number of random data samples produced
        data_size = config["data_set.random_dataset"]["size"]
        if not isinstance(data_size, int):
            raise ValueError(
                f"Expected integer value for `config['data_set.random_dataset']['size']`, but got {data_size}"
            )

        # The shape of each random data sample as a tuple.
        # i.e. (3, 29, 29) = 3 layers of 2d data, each layer is 29x29 elements.
        data_shape = tuple(config["data_set.random_dataset"]["shape"])
        if not len(data_shape):
            raise ValueError(
                "Expected `config['data_set.random_dataset']['data_shape']` to have at least 1 value."
            )

        for e in data_shape:
            if e < 1:
                raise ValueError(
                    f"Expected all values in `config['data_set.random_dataset']['data_shape']`\
                        to be > 0, but got {data_shape}."
                )
            if not isinstance(e, int):
                raise ValueError(
                    f"Expected all values in `config['data_set.random_dataset']['data_shape']`\
                        to be integers, but got {data_shape}."
                )

        # Random seed to use for reproducibility
        seed = config["data_set.random_dataset"]["seed"]
        rng = np.random.default_rng(seed)

        # Note: We raise exceptions if data_size is not an int, so we can assume
        # that turning that into a tuple and adding `data_shape` should work.
        self.data = rng.random((data_size,) + data_shape, np.float32)

        # Start our IDs at a random integer between 0 and 100
        id_start = rng.integers(100)
        self.id_list = list(range(id_start, id_start + data_size))

        # Randomly insert flawed values (np.nan, np.inf, -np.inf, None, other float)
        num_invalid_values = config["data_set.random_dataset"]["number_invalid_values"]
        if num_invalid_values:
            # Determine what value to use for invalid values
            invalid_value_type = config["data_set.random_dataset"]["invalid_value_type"]
            if isinstance(invalid_value_type, str):
                try:
                    invalid_value = INVALID_VALUES[invalid_value_type.lower()]
                except KeyError as err:
                    raise ValueError(
                        f"Invalid value type '{invalid_value_type}' provided. "
                        f"Expected `config['data_set.random_dataset']['invalid_value_type']` "
                        f"to beone of {list(INVALID_VALUES.keys())}"
                    ) from err
            else:
                if not isinstance(invalid_value_type, float):
                    raise ValueError(
                        f"Expected `config['data_set.random_dataset']['invalid_value_type']` to be "
                        f"a string or a float, but got {type(invalid_value_type)}."
                    )
                invalid_value = invalid_value_type
            flattened = np.ravel(self.data)
            random_inds = rng.choice(flattened.size, size=40, replace=False)
            flattened[random_inds] = invalid_value

        # If a list of possible labels is provided, create the random label list.
        self.provided_labels = config["data_set.random_dataset"]["provided_labels"]
        if self.provided_labels:
            self.labels = rng.choice(self.provided_labels, size=data_size)

        # Create a metadata_table that is used when visualizing data
        metadata_table = Table({"object_id": np.array(list(self.ids()))})

        super().__init__(config, metadata_table)


class HyraxRandomDataset(HyraxRandomDatasetBase, HyraxDataset, Dataset):
    """A map-style dataset yielding random numpy arrays."""

    def __getitem__(self, idx):
        """Return a torch.Tensor object at the index"""

        ret = {
            "index": idx,
            "object_id": self.id_list[idx],
            "image": from_numpy(self.data[idx]),
        }

        if self.provided_labels:
            ret["label"] = self.labels[idx]

        return ret

    def __len__(self):
        """Get the total number of samples in this dataset"""
        return len(self.data)

    def ids(self):
        """Yield IDs for the dataset"""
        for id_item in self.id_list:
            yield str(id_item)


class HyraxRandomIterableDataset(HyraxRandomDatasetBase, HyraxDataset, IterableDataset):
    """An iterable version of RandomDataset.

    Note: while ids will be generated automatically, calling the `ids` method of
    this class will simply return the index of the data."""

    def __iter__(self):
        for idx, image in enumerate(self.data):
            ret = {
                "index": idx,
                "object_id": idx,
                "image": from_numpy(image),
            }

            if self.provided_labels:
                ret["label"] = self.labels[idx]

            yield ret
