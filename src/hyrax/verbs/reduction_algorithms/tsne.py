from .algorithm_registry import ReductionAlgorithm


class TSNE(ReductionAlgorithm):
    """Placeholder TSNE reduction algorithm."""

    def __init__(self, config: dict, reduction_results=None):
        super().__init__(config, reduction_results)

    def fit(self, data):
        """Fit the TSNE model to the data."""
        raise NotImplementedError("TSNE reduction is not implemented yet.")

    def transform(self, data):
        """Transform the data with the fitted TSNE model."""
        raise NotImplementedError("TSNE reduction is not implemented yet.")
