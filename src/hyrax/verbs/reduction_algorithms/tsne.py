from .algorithm_registry import ReductionAlgorithm


class TSNE(ReductionAlgorithm):
    """Placeholder TSNE reduction algorithm."""

    def __init__(self, config: dict, reduction_results=None):
        super().__init__(config, reduction_results)
        self.reducer = None  # Placeholder for the actual TSNE model

    def transform(self, args: dict, num_batches: int):
        """Transform the data with the fitted TSNE model."""
        raise NotImplementedError("TSNE reduction is not implemented yet.")
