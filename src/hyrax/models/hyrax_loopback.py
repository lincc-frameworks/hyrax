import logging

import torch.nn as nn

from .model_registry import hyrax_model

logger = logging.getLogger()


@hyrax_model
class HyraxLoopback(nn.Module):
    """Simple model for testing which returns its own input"""

    def __init__(self, config, data_sample=None):
        from functools import partial

        super().__init__()
        # The optimizer needs at least one weight, so we add a dummy module here
        self.unused_module = nn.Linear(1, 1)
        self.config = config

        def load(self, weight_file):
            """Boilerplate function to load weights. However, this model has no
            weights so we do nothing."""
            pass

        # We override this way rather than defining a method because
        # Torch has some __init__ related cleverness which stomps our
        # load definition when performed in the usual fashion.
        self.load = partial(load, self)

    def forward(self, x):
        """We simply return our input"""
        if isinstance(x, tuple):
            # if x is a tuple, extract the first element (it should be a tensor)
            x, _ = x
        return x

    def train_step(self, batch):
        """Training is a noop"""
        logger.debug(f"Batch length: {len(batch)}")
        return {"loss": 0.0}
