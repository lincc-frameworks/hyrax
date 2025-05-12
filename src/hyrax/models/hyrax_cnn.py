# ruff: noqa: D101, D102

# This example model is taken from the PyTorch CIFAR10 tutorial:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa N812

from .model_registry import hyrax_model

logger = logging.getLogger(__name__)


@hyrax_model
class HyraxCNN(nn.Module):
    """
    This CNN is designed to work with datasets that are prepared with Hyrax's HSC Data Set class.
    """

    def __init__(self, config, shape):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.config = config

    def forward(self, x):
        # This check is inefficient - we assume that the example CNN will be primarily
        # used with the CIFAR10 dataset. During training, the `train_step` method
        # will unpack the tuple and only pass the first element to the `forward` method.
        # But for inference the entire tuple is passed in, so we need to handle
        # both cases.
        if isinstance(x, tuple):
            x, _ = x

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_step(self, batch):
        """This function contains the logic for a single training step. i.e. the
        contents of the inner loop of a ML training process.

        Parameters
        ----------
        batch : tuple
            A tuple containing the inputs and labels for the current batch.

        Returns
        -------
        Current loss value : dict
            Dictionary containing the loss value for the current batch.
        """
        inputs, labels = batch

        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}
