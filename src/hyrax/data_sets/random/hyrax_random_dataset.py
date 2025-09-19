import numpy as np
from astropy.table import Table
from torch.utils.data import Dataset, IterableDataset

from hyrax.data_sets.data_set_registry import HyraxDataset

INVALID_VALUES = {
    "nan": np.nan,
    "inf": np.inf,
    "-inf": -np.inf,
    "none": None,
}
"""Mapping of string representation of invalid values to numpy representations."""


class HyraxRandomDatasetBase:
    """
    This is the base class for the random datasets provided by Hyrax.

    .. warning::

        Direct use of ``HyraxRandomDatasetBase`` is not advised. When working
        with Hyrax, prefer to use ``HyraxRandomDataset`` or ``HyraxRandomIterableDataset``.
    """

    data: np.ndarray
    """The random data samples produced by the dataset."""
    id_list: list
    """A list of sequential numeric IDs for each data sample."""
    provided_labels: list
    """A list of labels randomly selected from the provided list of possible labels."""

    def __init__(self, config, data_location):
        """
        .. py:method:: __init__(config, data_location)

        Initialize the dataset using the parameters defined in the configuration.

        Parameter included for API consistency with other dataset classes, though
        not used by this implementation. All parameters are controlled by the following
        keys under the ``["data_set"]["HyraxRandomDataset"]`` table in the configuration:

        - ``size``: The number of random data samples to produce.
        - ``shape``: The shape of each random data sample as a tuple (e.g. (3, 29, 29) = 3
          layers of 2D data, each layer is 29x29 elements).
        - ``seed``: The random seed to use for reproducibility.
        - ``provided_labels``: A list of possible labels to randomly select from.
          If this is provided, the dataset will randomly select a label for each data sample.
        - ``number_invalid_values``: The number of invalid values to insert into the data.
        - ``invalid_value_type``: The type of invalid value to insert into the data.
          Valid values are "nan", "inf", "-inf", "none", or a float value.
        """
        # The total number of random data samples produced
        data_size = config["data_set"]["HyraxRandomDataset"]["size"]
        if not isinstance(data_size, int):
            raise ValueError(
                f"Expected integer for `config['data_set']['random_dataset']['size']`, but got {data_size}"
            )

        # The shape of each random data sample as a tuple.
        # i.e. (3, 29, 29) = 3 layers of 2d data, each layer is 29x29 elements.
        data_shape = tuple(config["data_set"]["HyraxRandomDataset"]["shape"])
        if not len(data_shape):
            raise ValueError(
                "Expected `config['data_set']['random_dataset']['data_shape']` to have at least 1 value."
            )

        for e in data_shape:
            if e < 1:
                raise ValueError(
                    f"Expected all values in `config['data_set']['random_dataset']['data_shape']`\
                        to be > 0, but got {data_shape}."
                )
            if not isinstance(e, int):
                raise ValueError(
                    f"Expected all values in `config['data_set']['random_dataset']['data_shape']`\
                        to be integers, but got {data_shape}."
                )

        # Random seed to use for reproducibility
        seed = config["data_set"]["HyraxRandomDataset"]["seed"]
        rng = np.random.default_rng(seed)

        # Note: We raise exceptions if data_size is not an int, so we can assume
        # that turning that into a tuple and adding `data_shape` should work.
        self.data = rng.random((data_size,) + data_shape, np.float32)

        # Start our IDs at a random integer between 0 and 100
        id_start = rng.integers(100)
        self.id_list = np.arange(id_start, id_start + data_size)

        # Randomly insert flawed values (np.nan, np.inf, -np.inf, None, other float)
        num_invalid_values = config["data_set"]["HyraxRandomDataset"]["number_invalid_values"]
        if num_invalid_values:
            # Determine what value to use for invalid values
            invalid_value_type = config["data_set"]["HyraxRandomDataset"]["invalid_value_type"]
            if isinstance(invalid_value_type, str):
                try:
                    invalid_value = INVALID_VALUES[invalid_value_type.lower()]
                except KeyError as err:
                    raise ValueError(
                        f"Invalid value type '{invalid_value_type}' provided. "
                        f"Expected `config['data_set']['random_dataset']['invalid_value_type']` "
                        f"to be one of {list(INVALID_VALUES.keys())}"
                    ) from err
            else:
                if not isinstance(invalid_value_type, float):
                    raise ValueError(
                        f"Expected `config['data_set']['random_dataset']['invalid_value_type']` to be "
                        f"a string or a float, but got {type(invalid_value_type)}."
                    )
                invalid_value = invalid_value_type
            flattened = np.ravel(self.data)
            random_inds = rng.choice(flattened.size, size=40, replace=False)
            flattened[random_inds] = invalid_value

        # If a list of possible labels is provided, create the random label list.
        self.provided_labels = config["data_set"]["HyraxRandomDataset"]["provided_labels"]
        if self.provided_labels:
            self.labels = rng.choice(self.provided_labels, size=data_size)

        # Create a metadata_table that is used when visualizing data
        metadata_table = Table(
            {
                "object_id": np.array([str(id) for id in self.id_list]),
                "meta_field_1": np.array(list(range(data_size, 0, -1))) / 2,
                "meta_field_2": np.array(list(range(data_size, 0, -1))) / 3,
            },
        )

        super().__init__(config, metadata_table)

        self.data_location = data_location

    def get_image(self, idx: int) -> np.ndarray:
        """Get the image at the given index as a NumPy array."""
        return self.data[idx]

    def get_label(self, idx: int) -> str:
        """Get the label at the given index."""
        if self.provided_labels:
            return self.labels[idx]
        return None

    def get_object_id(self, idx: int) -> str:
        """Get the index of the item."""
        return str(self.id_list[idx])


class HyraxRandomDataset(HyraxRandomDatasetBase, HyraxDataset, Dataset):
    """This dataset is stand-in for a map-style dataset.
    It will produce random numpy arrays along with sequential numeric ids and,
    optionally, labels randomly selected from the provided list of possible labels.
    """

    def __getitem__(self, idx: int) -> dict:
        """Get a data sample by index. The returned dictionary will contain the
        following keys:

        - ``index``: The index of the data sample.
        - ``object_id``: The ID of the data sample.
        - ``image``: The data sample as a numpy array.
        - ``label``: The label of the data sample (if provided).

        Parameters
        ----------
        idx : int
            The index of the data sample to retrieve.

        Returns
        -------
        dict
            A dictionary containing the data sample and its metadata.

        """

        ret = {
            "data": {
                "index": idx,
                "object_id": self.get_object_id(idx),
                "image": self.get_image(idx),
            },
            "object_id": self.get_object_id(idx),
        }

        if self.provided_labels:
            ret["data"]["label"] = self.get_label(idx)

        return ret

    def __len__(self):
        """Get the total number of samples in this dataset. This should be return
        the same value as the `size` parameter in the configuration."""
        return len(self.data)

    def ids(self):
        """This function yields IDs for the dataset. It can be used as an iterable
        in a loop, or converted to a list by wrapping the function call in ``list(...)``."""
        for id_item in self.id_list:
            yield str(id_item)


class HyraxRandomIterableDataset(HyraxRandomDatasetBase, HyraxDataset, IterableDataset):
    """This dataset is stand-in for a iterable-style, or streaming, dataset.
    It will produce random numpy arrays and, optionally, labels randomly
    selected from the provided list of possible labels.

    .. note::

        While ids will be generated automatically for this dataset, calling the
        ``ids`` method of this dataset will return the index instead of the id.
    """

    def __iter__(self):
        """Yield the next data sample. The returned dictionary will have the
        following form:

        - ``data``: A dictionary containing the following keys:
        -- ``index``: The index of the data sample.
        -- ``object_id``: The value will be the same as ``index`` for this dataset.
        -- ``image``: The data sample as a numpy array.
        -- ``label``: The label of the data sample (if provided).

        Returns
        -------
        dict
            A dictionary containing a data sample and its metadata.

        """
        for idx, _ in enumerate(self.data):
            ret = {
                "data": {
                    "index": idx,
                    "object_id": self.get_object_id(idx),
                    "image": self.get_image(idx),
                },
                "object_id": self.get_object_id(idx),
            }

            if self.provided_labels:
                ret["data"]["label"] = self.get_label(idx)

            yield ret
