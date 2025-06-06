import torch.nn as nn

from .model_registry import hyrax_model


@hyrax_model
class HyraxLoopback(nn.Module):
    """Simple model for testing which returns its own input"""

    def __init__(self, config, shape):
        from functools import partial

        super().__init__()
        # This is created so the optimizer can find at least one weight
        self.unused_module = nn.Conv2d(1, 1, kernel_size=1, stride=0, padding=0)
        self.config = config

        def load(self, weight_file):
            """Load Weights, we have no weights so we do nothing"""
            pass

        # We override this way rather than defining a method because
        # Torch has some __init__ related cleverness which stomps our
        # load definition when performed in the usual fashion.
        self.load = partial(load, self)

    def forward(self, x):
        """We simply return our input"""
        if isinstance(x, tuple):
            x, label = x
        return x

    def train_step(self, batch):
        """Training is a noop"""
        return {"loss": 0.0}
