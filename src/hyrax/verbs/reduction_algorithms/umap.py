import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import psutil
import umap

from hyrax.datasets.result_dataset import ResultDatasetWriter

from .algorithm_registry import ReductionAlgorithm

logger = logging.getLogger(__name__)


class UMAP(ReductionAlgorithm):
    """UMAP reduction implementation."""

    def __init__(self, config: dict, reduction_results=None):
        super().__init__(config, reduction_results)
        self.reducer = umap.UMAP(**self.config["reduce"]["umap"]["kwargs"])

    def save_model(self, results_dir: Path):
        """
        Save the fitted UMAP model to a pickle file.

        Parameters
        ----------
        results_dir : Path
            The directory where the model should be saved.
            The model will be saved as 'umap.pickle' in this directory.
        """
        with open(results_dir / "umap.pickle", "wb") as f:
            pickle.dump(self.reducer, f)

    def load_model(self, expected_input_dim: int, model_path: Union[Path, str] | None = None):
        """
        Load a pre-existing UMAP model from disk.

        Parameters
        ----------
        expected_input_dim : int
            The expected number of input features for the loaded model.
        model_path : Path or str, optional
            The path to the file to load the model from.
            If not specified, method will look in the config for a default model path.

        Returns
        -------
        UMAP
            The UMAP instance with the loaded model.
        """
        if model_path is None:
            model_path = self.config["reduce"]["model_path"]

        if not model_path:
            logger.info("No pre-existing UMAP model found. A new model will be fitted.")
            return None

        # Path validity check
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"UMAP model file not found: {model_path}")

        logger.info(f"Loading pre-existing UMAP model from {model_path}")
        reducer = self._load_pickle(model_path)

        # UMAP type check
        if not isinstance(reducer, umap.UMAP):
            raise ValueError(f"The loaded model is not a UMAP instance: {type(reducer)}")

        # Input feature dim check
        if reducer._raw_data.shape[1] != expected_input_dim:
            raise ValueError(
                f"The input dimension of the loaded UMAP model ({reducer._raw_data.shape[1]})"
                f" does not match the dimension of the inference data ({expected_input_dim})."
            )

        # Output dim check
        if reducer.n_components != self.reducer.n_components:
            raise ValueError(
                f"The output dimension of the loaded UMAP model ({reducer.n_components})"
                f" does not match the configured n_components ({self.reducer.n_components})."
            )

        self.reducer = reducer

    def fit(self, data_sample: np.ndarray):
        """
        Fit the UMAP model to a sample of inference data. The fitted model is stored in
        the instance variable `self.reducer` and can be used for transforming data.

        Parameters
        ----------
        data_sample : numpy.ndarray
            The data sample used to fit the model.
        """
        self._log_memory_usage("Before fitting umap")
        logger.info("Fitting the UMAP")
        self.reducer.fit(data_sample)
        self._log_memory_usage("After fitting umap")

    def transform(self, args: dict, num_batches: int, reduction_results: ResultDatasetWriter):
        """
        Transform data with a fitted UMAP model. Use parallel processing if specified in the config.

        Parameters
        ----------
        args : dict
            A dictionary containing the data to be transformed.

        num_batches : int
            The total number of batches that the data is split into for transformation.

        reduction_results : ResultDatasetWriter
            An instance of ResultDatasetWriter where the transformed results should be written.
        """
        if self.reducer is None:
            raise RuntimeError("Cannot transform data before loading or fitting a UMAP model.")

        from tqdm.auto import tqdm

        # logger.info("Transforming data with UMAP")
        # return self.reducer.transform(data)
        # if self.config["reduce"]["umap"].get("parallel", False):
        if self.config["reduce"]["umap"]["parallel"]:
            import multiprocessing as mp

            # Process pool loop
            # Use 'spawn' context to safely create subprocesses after
            # OpenMP threads are being opened by other processes in hyrax
            # Not using spawn causes the issue linked below
            # https://github.com/lincc-frameworks/hyrax/issues/291
            # TODO: Find more elegant solution than just using spawn
            with mp.get_context("spawn").Pool(processes=mp.cpu_count()) as pool:
                for batch_ids, transformed_batch in tqdm(
                    pool.imap(self._transform_batch, args),
                    desc="Creating lower dimensional representation using UMAP:",
                    total=num_batches,
                ):
                    reduction_results.write_batch(batch_ids, transformed_batch)
        else:
            # Sequential loop
            for batch_ids, batch in tqdm(
                args,
                desc="Creating lower dimensional representation using UMAP:",
                total=num_batches,
            ):
                transformed_batch = self.reducer.transform(batch)
                self._log_memory_usage(f"During transformation of batch of shape {batch.shape}")
                reduction_results.write_batch(batch_ids, transformed_batch)

        reduction_results.commit()  # Ensure all data is written and finalized

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
        """Private helper to transform a single batch

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
            second element is the results of running the umap transform on the input as a numpy array.
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
