import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import psutil

from hyrax.plugin_utils import update_registry

logger = logging.getLogger(__name__)
ALGORITHM_REGISTRY: dict[str, type["ReductionAlgorithm"]] = {}


class ReductionAlgorithm:
    """Abstract base class for all reduction algorithms."""

    from hyrax.datasets.result_dataset import ResultDatasetWriter

    def __init__(self, config: dict, reduction_results: ResultDatasetWriter | None = None):
        self._config = config
        self._reduction_results = reduction_results
        self.reducer = None

    @property
    def config(self):
        """Return the configuration dictionary for this reduction algorithm."""
        return self._config

    @property
    def reduction_results(self):
        """Return the result dataset writer for this reduction algorithm."""
        return self._reduction_results

    def __init_subclass__(cls):
        from abc import ABC

        if ABC in cls.__bases__:
            return

        # We only require a user to implement a transform method.
        if not hasattr(cls, "transform"):
            raise RuntimeError(
                f"Reduction algorithm {cls.__name__} is missing required transform function. "
                "transform must be defined."
            )

        # Ensure the class is in the registry so the config system can find it
        update_registry(ALGORITHM_REGISTRY, cls.__name__.lower(), cls)

    def fit(self, data_sample: np.ndarray):
        """
        Fit the reduction algorithm to the data.
        Set the internal state of the reducer based on the provided data sample.

        Parameters
        ----------
        data_sample : numpy.ndarray
            The data sample used to fit the model.
        """
        logger.info("Independent fit method not applicable for this reducer")
        pass

    def transform(self, args: dict, num_batches: int):
        """
        Transform the data with a fitted reducer.

        Parameters
        ----------
        args : dict
            A dictionary containing the data to be transformed.

        num_batches : int
            The total number of batches that the data is split into for transformation.
        """
        raise NotImplementedError("Subclasses must implement the transform method.")

    def save_model(self, model_path: Union[Path, str] | None = None):
        """
        Save the reducer model to a picklefile.

        Parameters
        ----------
        model_path : Path or str
            The path to save the model to.
        """
        logger.info("Model saving not applicable for this reducer")
        pass

    def load_model(self, expected_input_dim: int, model_path: Union[Path, str] | None = None):
        """
        Load the reducer model from a file.

        Parameters
        ----------
        expected_input_dim : int
            The expected number of input features for the loaded model.
        model_path : Path or str, optional
            The path to the file to load the model from.

        Returns
        -------
        ReductionAlgorithm
            The reduction algorithm instance with the loaded model.
        """
        logger.info("Model loading not applicable for this reducer")
        pass

    def _load_pickle(self, model_path: Union[Path, str]):
        """
        Helper function to wrap loading a pickle file from a given path for easier testing.

        Parameters
        ----------
        model_path : str or Path
            The file path to the pickle file.

        Returns
        -------
        object
            The object loaded from the pickle file.
        """
        model_path = Path(model_path)
        with open(model_path, "rb") as f:
            object = pickle.load(f)
            return object

    def _transform_batch(self, batch_tuple: tuple):
        """Private helper to transform a single batch with fitted reducer.

        Parameters
        ----------
        batch_tuple : tuple()
            first element is the IDs of the batch as a numpy array
            second element is the inference results to transform as a numpy array with shape (batch_len, N)
            where N is the total number of dimensions in the inference result. Caller flattens all inference
            result axes for us.

        Returns
        -------
        tuple
            first element is the ids of the batch as a numpy array
            second element is the results of running the transform on the input as a numpy array.
        """
        batch_ids, batch = batch_tuple
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            logger.debug("Transforming a batch ...")
            return (batch_ids, self.reducer.transform(batch))

    @staticmethod
    def _log_memory_usage(message: str = ""):
        """
        Log the current resident set size (RSS) memory usage of the current process in gigabytes.

        Parameters
        ----------
        message : str, optional
            A descriptive message to include in the log output for context.

        Notes
        -----
        This method is intended for debugging and performance monitoring.
        """
        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / 1024**3
        logger.debug(f"{message} | Memory usage: {mem_gb:.2f} GB")


def is_reducer_class(cli_name: str) -> bool:
    """
    Returns true if the reducer algorithm has a class based implementation

    Parameters
    ----------
    cli_name : str
        The name of the reducer algorithm on the command line interface

    Returns
    -------
    bool
        True if the reducer algorithm has a class-based implementation
    """
    return cli_name in ALGORITHM_REGISTRY and ALGORITHM_REGISTRY.get(cli_name) is not None


def fetch_reducer_class(cli_name: str) -> type[ReductionAlgorithm]:
    """
    Fetch the class implementing the reducer algorithm specified.
    The class must be a subclass of ReductionAlgorithm and must be registered in the ALGORITHM_REGISTRY.

    Parameters
    ----------
    cli_name : str
        The name of the reducer algorithm on the command line interface

    Returns
    -------
    type[ReductionAlgorithm]
        The class implementing the reducer algorithm.
    """
    reducer_cls = ALGORITHM_REGISTRY.get(cli_name.lower())
    if not reducer_cls:
        raise KeyError(
            f"Reduction algorithm {cli_name} is not registered. "
            f"Available algorithms: {sorted(ALGORITHM_REGISTRY)}"
        )

    return reducer_cls
