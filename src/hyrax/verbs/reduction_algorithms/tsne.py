import logging
import warnings

import sklearn.manifold as sklearn_manifold

from .algorithm_registry import ReductionAlgorithm

logger = logging.getLogger(__name__)


class TSNE(ReductionAlgorithm):
    """TSNE reduction implementation."""

    def __init__(self, config: dict, reduction_results=None):
        super().__init__(config, reduction_results)
        self.reducer = sklearn_manifold.TSNE(**self.config["reduce"]["tsne"]["kwargs"])

    def save_model(self, _):
        """
        TSNE does not support saving the model. This method is a no-op.
        """
        logger.warning("TSNE does not support saving the model. Skipping save_model.")

    def load_model(self, _):
        """
        TSNE does not support loading a pre-existing model. This method is a no-op.
        """
        logger.warning("TSNE does not support loading a pre-existing model. Skipping load_model.")

    def fit(self, _):
        """
        TSNE does not support a separate fitting stage. This method is a no-op.
        """
        logger.warning("TSNE does not support a separate fitting stage. Skipping fit.")

    def transform(self, args: dict, num_batches: int):
        """
        Fit and transform data with TSNE model.

        Parameters
        ----------
        args : dict
            A dictionary containing the data to be transformed.

        num_batches : int
            The total number of batches that the data is split into for transformation.
        """
        from tqdm.auto import tqdm

        if self.config["reduce"]["parallel"]:
            import multiprocessing as mp

            # Process pool loop
            # Use 'spawn' context to safely create subprocesses after
            # OpenMP threads are being opened by other processes in hyrax
            # Not using spawn causes the issue linked below
            # https://github.com/lincc-frameworks/hyrax/issues/291
            # TODO: Find more elegant solution than just using spawn
            with mp.get_context("spawn").Pool(processes=mp.cpu_count()) as pool:
                for batch_ids, transformed_batch in tqdm(
                    pool.imap(self._fit_transform_batch, args),
                    desc="Creating lower dimensional representation using t-SNE:",
                    total=num_batches,
                ):
                    self.reduction_results.write_batch(batch_ids, transformed_batch)
        else:
            # Sequential loop
            for batch_ids, batch in tqdm(
                args,
                desc="Creating lower dimensional representation using t-SNE:",
                total=num_batches,
            ):
                transformed_batch = self.reducer.fit_transform(batch)
                self._log_memory_usage(f"During transformation of batch of shape {batch.shape}")
                self.reduction_results.write_batch(batch_ids, transformed_batch)

        self.reduction_results.commit()

    def _fit_transform_batch(self, batch_tuple: tuple):
        """Private helper to fit_transform a single batch

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
            second element is the results of running the tsne transform on the input as a numpy array.
        """
        batch_ids, batch = batch_tuple
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            logger.debug("Transforming a batch ...")
            return (batch_ids, self.reducer.fit_transform(batch))
