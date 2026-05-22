from .algorithm_registry import ReductionAlgorithm


class PCA(ReductionAlgorithm):
    """Placeholder PCA reduction algorithm."""

    def __init__(self, config: dict, reduction_results=None):
        super().__init__(config, reduction_results)

    def fit(self, data):
        """Fit the PCA model to the data."""
        raise NotImplementedError("PCA reduction is not implemented yet.")

    def transform(self, data):
        """Transform the data with the fitted PCA model."""
        raise NotImplementedError("PCA reduction is not implemented yet.")
