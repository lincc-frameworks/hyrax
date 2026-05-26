from pathlib import Path
from typing import Union

from hyrax.datasets.result_dataset import ResultDatasetWriter

from .algorithm_registry import ReductionAlgorithm


class PCA(ReductionAlgorithm):
    """Placeholder PCA reduction algorithm."""

    def __init__(self, config: dict, reduction_results=None):
        super().__init__(config, reduction_results)
        self.reducer = None  # Placeholder for the actual PCA model

    def save_model(self, results_dir: Path):
        """Save the fitted PCA model to a pickle file."""
        raise NotImplementedError("PCA reduction is not implemented yet.")

    def load_model(self, expected_input_dim: int, model_path: Union[Path, str] | None = None):
        """Load a pre-existing PCA model from disk."""
        raise NotImplementedError("PCA reduction is not implemented yet.")

    def fit(self, data):
        """Fit the PCA model to the data."""
        raise NotImplementedError("PCA reduction is not implemented yet.")

    def transform(self, args: dict, num_batches: int, reduction_results: ResultDatasetWriter):
        """Transform the data with the fitted PCA model."""
        raise NotImplementedError("PCA reduction is not implemented yet.")
