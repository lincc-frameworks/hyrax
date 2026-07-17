# ruff: noqa: D102, B027
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from hyrax.plugin_utils import get_or_load_class, update_registry

logger = logging.getLogger(__name__)
DATASET_REGISTRY: dict[str, type["HyraxDataset"]] = {}


class HyraxDataset:
    """
    How to make a hyrax dataset:

    .. code-block:: python

        from hyrax.datasets import HyraxDataset

        class MyDataset(HyraxDataset):
            def __init__(self, config: dict):
                super().__init__(config)

            def __len__(self):
                # Your len function goes here
                pass

    Further documentation is in the :doc:`/pre_executed/external_dataset_class` example notebook.

    """

    def __init__(self, config: dict):
        """
        .. py:method:: __init__

        Overall initialization for all Datasets which saves the config

        Subclasses of HyraxDataset ought call this at the end of their __init__ like:

        .. code-block:: python

            from hyrax.datasets import HyraxDataset

            class MyDataset(HyraxDataset):
                def __init__(config):
                    <your code>
                    super().__init__(config)

        Parameters
        ----------
        config : dict
            The runtime configuration for hyrax
        """

        self._config = config

    @property
    def config(self):
        return self._config

    def __init_subclass__(cls):
        from abc import ABC

        if ABC in cls.__bases__:
            return

        # We only require a user to implement a __len__ method.
        if not hasattr(cls, "__len__"):
            msg = f"Hyrax data set {cls.__name__} is missing required length function. "
            msg += "__len__ must be defined."
            raise RuntimeError(msg)

        # Ensure the class is in the registry so the config system can find it
        update_registry(DATASET_REGISTRY, cls.__name__, cls)

    def augment_cache_key(self, idx: int, rng_seed: np.int64) -> np.int64 | None:
        """Return a cache key for augmented data, or None to skip caching.

        Base (non-augmented) data is always cached by index. This method is
        only called when augmentation is active, to decide whether the
        augmented result should also be cached. The default returns ``None``
        (augmented data is regenerated each access), which is the standard
        expectation in ML training.

        Override this when augmented results are deterministic and expensive
        to recompute.

        Parameters
        ----------
        idx : int
            The dataset-local index.
        rng_seed : np.int64
            The rng_seed passed to ``augment_<field>`` methods.

        Returns
        -------
        np.int64 | None
            Cache key, or ``None`` to skip caching augmented data.
        """
        return None

    def on_epoch_start(self, verb: str):
        """Called at the beginning of each epoch (or once for single-pass verbs).

        Override in subclasses to respond to epoch-level lifecycle events.

        Parameters
        ----------
        verb : str
            Name of the verb that is running, e.g. ``"train"``, ``"infer"``,
            ``"test"``, or ``"engine"``.
        """
        pass


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
        raise RuntimeError("dataset_class must be specified in 'data_request'.")

    dataset_cls = get_or_load_class(class_name, DATASET_REGISTRY)

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

        data_transformed = self.transform(data_torch) if self.transform is not None else data_torch

        return data_transformed.numpy()

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
