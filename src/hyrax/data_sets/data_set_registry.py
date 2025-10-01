# ruff: noqa: D102, B027
import logging
from collections.abc import Generator
from types import MethodType
from typing import Any, Callable

import numpy.typing as npt

from hyrax.config_utils import ConfigDict
from hyrax.plugin_utils import get_or_load_class, update_registry

logger = logging.getLogger(__name__)
DATA_SET_REGISTRY: dict[str, type["HyraxDataset"]] = {}


class HyraxDataset:
    """
    How to make a hyrax dataset:

    .. code-block:: python

        from hyrax.data_sets import HyraxDataset
        from torch.utils.data import Dataset

        class MyDataset(HyraxDataset, Dataset):
            def __init__(self, config: dict):
                super().__init__(config)

            def __getitem__():
                # Your getitem goes here
                pass

            def __len__ ():
                # Your len function goes here
                pass

    Optional interfaces:

    ``ids()`` -> Subclasses may override this directly with their own ids function
    returning a generator of strings

    ``metadata`` -> Subclasses may pass an astropy table of metadata to ``__init__`` in the
    superclass. This table of metadata will be available through the ``metadata_fields`` and
    ``metadata`` functions.  If desired, a subclass may override these functions directly
    rather than using the astropy Table interface.

    Further documentation is in the :doc:`/pre_executed/custom_dataset` example notebook.

    """

    def __init__(self, config: ConfigDict, metadata_table=None):
        """
        .. py:method:: __init__

        Overall initialization for all DataSets which saves the config

        Subclasses of HyraxDataSet ought call this at the end of their __init__ like:

        .. code-block:: python

            from hyrax.data_sets import HyraxDataset
            from torch.utils.data import Dataset

            class MyDataset(HyraxDataset, Dataset):
                def __init__(config):
                    <your code>
                    super().__init__(config)

        If per tensor metadata is available, it is recommended that dataset authors create an
        astropy Table of that data, in the same order as their data and pass that `metadata_table`
        as shown below:

        .. code-block:: python

            from hyrax.data_sets import HyraxDataset
            from torch.utils.data import Dataset
            from astropy.table import Table

            class MyDataset(HyraxDataset, Dataset):
                def __init__(config):
                    <your code>
                    metadata_table = Table(<Your catalog data goes here>)
                    super().__init__(config, metadata_table)

        Parameters
        ----------
        config : ConfigDict, Optional
            The runtime configuration for hyrax
        metadata_table : Optional[Table], optional
            An Astropy Table with
            1. the metadata columns desired for visualization AND
            2. in the order your data will be enumerated.
        """
        import numpy as np

        self._config = config
        self._metadata_table = metadata_table

        # If your metadata does not contain an object_id field
        # we use your required .ids() method to create the column
        if self._metadata_table is not None:
            colnames = self._metadata_table.colnames
            if "object_id" not in colnames:
                # Note: See https://github.com/lincc-frameworks/hyrax/issues/374
                # for iterable dataset support discussion.
                ids = np.array(list(self.ids()))
                self._metadata_table.add_column(ids, name="object_id")

            def _make_getter(column):
                def getter(self, idx, _col=column):
                    return self._metadata_table[_col][idx]

                return getter

            for col in self._metadata_table.colnames:
                method_name = f"get_{col}"
                if not hasattr(self, method_name):
                    setattr(self, method_name, MethodType(_make_getter(col), self))

        self.tensorboardx_logger = None

    @classmethod
    def is_iterable(cls):
        """
        Returns true if underlying dataset is iterable style, supporting __iter__ vs map style
        where  __getitem__/__len__ are the preferred access methods.

        Returns
        -------
        bool
            True if underlying dataset is iterable
        """
        from torch.utils.data import Dataset, IterableDataset

        if issubclass(cls, (Dataset, IterableDataset)):
            # All torch IterableDatasets are also Datasets
            return issubclass(cls, IterableDataset)
        else:
            return hasattr(cls, "__iter__")

    @classmethod
    def is_map(cls):
        """
        Returns true if underlying dataset is map style, supporting __getitem__/__len__ vs iterable
        where __iter__ is the preferred access method.

        Returns
        -------
        bool
            True if underlying dataset is map-style
        """
        from torch.utils.data import Dataset, IterableDataset

        if issubclass(cls, (Dataset, IterableDataset)):
            # All torch IterableDatasets are also Datasets
            return not issubclass(cls, IterableDataset)
        else:
            return hasattr(cls, "__getitem__")

    @property
    def config(self):
        return self._config

    def __init_subclass__(cls):
        from abc import ABC

        if ABC in cls.__bases__:
            return

        # Paranoia. Deriving from a torch dataset class should ensure this, but if an external dataset author
        # Forgets to to do that, we tell them.
        if (not hasattr(cls, "__iter__")) and not (hasattr(cls, "__getitem__") and hasattr(cls, "__len__")):
            msg = f"Hyrax data set {cls.__name__} is missing required iteration functions."
            msg += "__len__ and __getitem__ (or __iter__) must be defined. It is recommended to derive from"
            msg += " torch.utils.data.Dataset (or torch.utils.data.IterableDataset) which will enforce this."
            raise RuntimeError(msg)

        # TODO?:If the subclass has __iter__ and not __getitem__/__len__ perhaps add an __iter__ with a
        #       warning Because to the extent the __getitem__/__len__ functions get used they'll exhaust the
        #       iterator and possibly remove any benefit of having them around.

        # TODO?:If the subclass has __getitem__/__len__ and not __iter__ add an __iter__. This is less
        #       dangerous, and should probably just be an info log.
        #
        #       This might be better as a function on this base class, but doing it here gives us an
        #       opportunity to do configuration or logging to help people navigate writing a dataset?

        # Ensure the class is in the registry so the config system can find it
        update_registry(DATA_SET_REGISTRY, cls.__name__, cls)

    def ids(self) -> Generator[str]:
        """This is the default IDs function you get when you derive from hyrax Dataset

        Returns
        -------
        Generator[str]
            A generator yielding all the string IDs of the dataset.

        """
        if self.is_map():
            for x in range(len(self)):
                yield str(x)
        elif self.is_iterable():
            for index, _ in enumerate(iter(self)):
                yield (str(index))
        else:
            raise NotImplementedError(
                f"Dataset class '{self.__class__.__name__}' must implement either "
                "__len__ and __getitem__ for map-style datasets, or __iter__ for "
                "iterable-style datasets to use automatic id() generation."
            )

    def sample_data(self) -> dict:
        """Get a sample from the dataset. This is a convenience function that returns
        the first sample from the dataset, regardless of whether it is iterable
        or map-style. Often this will be used to instantiate a model that adjusts
        its form based on the shape of the data."""

        if self.is_map():
            return self[0]
        elif self.is_iterable():
            return next(iter(self))
        else:
            raise NotImplementedError(
                "You must define __getitem__ or __iter__ to use the default `get_sample()` method."
            )

    def metadata_fields(self) -> list[str]:
        """Returns a list of metadata fields supported by this object

        Returns
        -------
        list[str]
            The column names of the metadata table passed. Empty string if no metadata was provided at
            during construction of the HyraxDataset (or derived class).
        """
        return [] if self._metadata_table is None else list(self._metadata_table.colnames)

    def metadata(self, idxs: npt.ArrayLike, fields: list[str]) -> npt.ArrayLike:
        """Returns a table representing the metadata given an array of indexes and a list of fields.

        Parameters
        ----------
        idxs : npt.ArrayLike
            The indexes of the relevant tensor objects
        fields : list[str]
            The names of the fields you would like returned. All values must be among those returned by
            metadata_fields()

        Returns
        -------
        npt.ArrayLike
            A numpy record array of your metadata, with only the columns specified.
            Roughly equivalent to: `metadata_table[idxs][fields].as_array()` where metadata_table is the
            astropy table that the HyraxDataset (or derived class) was constructed with.

        Raises
        ------
        RuntimeError
            When none of the provided fields are
        """
        metadata_fields = self.metadata_fields()
        for field in fields:
            if field not in metadata_fields:
                msg = f"Field {field} is not available for {self.__class__.__name__}."
                logger.error(msg)

        columns = [field for field in fields if field in metadata_fields]

        if len(columns) == 0:
            msg = (
                f"None of the metadata fields passed [{fields}] are available for {self.__class__.__name__}."
            )
            raise RuntimeError(msg)

        return self._metadata_table[idxs][columns].as_array()


def fetch_dataset_class(class_name: str) -> type[HyraxDataset]:
    """Fetch the dataset class from the registry.

    Parameters
    ----------
    class_name : str
        The name of the dataset class to fetch. Either the class name of a built
      in dataset, or the fully qualified name of a user-defined dataset.
      e.g. "my_module.my_submodule.MyDatasetClass" or "HyraxRandomDataset".

    Returns
    -------
    type[HyraxDataset]
        The dataset class.

    Raises
    ------
    ValueError
        If a built in dataset was requested, but not found in the registry.
    ValueError
        If no dataset was specified in the runtime configuration.
    """

    if not class_name:
        raise RuntimeError("dataset_class must be specified in 'model_inputs'.")

    dataset_cls = get_or_load_class(class_name, DATA_SET_REGISTRY)

    return dataset_cls


class HyraxImageDataset:
    """
    This is a mixin for Image datasets primarily concerned with providing utility functions to
    allow derived classes to set and apply transformations based on configs.

    The various set_*_transform functions stack individual transformations on a single stack

    The stack can be applied with apply_transform.
    """

    def set_function_transform(self):
        from torchvision.transforms.v2 import Lambda

        function_name = self.config["data_set"]["transform"]
        if function_name:
            transform_func = self._get_np_function(function_name)
            self._update_transform(Lambda(lambd=transform_func))

    def set_crop_transform(self, cutout_shape=None):
        from torchvision.transforms.v2 import CenterCrop

        if cutout_shape is None:
            cutout_shape = self.config["data_set"]["crop_to"] if self.config["data_set"]["crop_to"] else None

        if (not isinstance(cutout_shape, list) and not isinstance(cutout_shape, tuple)) or len(
            cutout_shape
        ) != 2:
            msg = "Must provide a cutout shape in config['data_set']['crop_to']."
            msg += " Shape should be a list of integer pixel sizes e.g. [100,100]"
            raise RuntimeError(msg)

        self._update_transform(CenterCrop(size=cutout_shape))

    def apply_transform(self, data_torch):
        if self.__dict__.get("transform", False) is False:
            self.transform = None

        return self.transform(data_torch) if self.transform is not None else data_torch

    def _update_transform(self, new_transform):
        from torchvision.transforms.v2 import Compose

        if self.__dict__.get("transform", False) is False:
            self.transform = None

        self.transform = new_transform if self.transform is None else Compose([new_transform, self.transform])

    def _get_np_function(self, transform_str: str) -> Callable[..., Any]:
        """
        _get_np_function. Returns the numpy mathematical function that the
        supplied string maps to; or raises an error if the supplied string
        cannot be mapped to a function.

        Parameters
        ----------
        transform_str: str
            The string to me mapped to a numpy function
        """
        import numpy as np

        try:
            func: Callable[..., Any] = getattr(np, transform_str)
            if callable(func):
                return func
        except AttributeError as err:
            msg = f"{transform_str} is not a valid numpy function.\n"
            msg += "The string passed to the transform variable needs to be a numpy function"
            raise RuntimeError(msg) from err
