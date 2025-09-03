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

    def __init__(self, config, shape=(3, 32, 32)):
        super().__init__()
        self.config = config

        self.num_input_channels, self.image_width, self.image_height = shape
        hidden_channels_1 = 6
        hidden_channels_2 = 16

        # Calculate how much our convolutional layers and pooling will affect
        # the size of final convolution.
        #
        # If the number of layers are changed this will need to be rewritten.
        conv1_end_w = self.conv2d_output_size(self.image_width, kernel_size=5)
        conv1_end_h = self.conv2d_output_size(self.image_height, kernel_size=5)

        pool1_end_w = self.pool2d_output_size(conv1_end_w, kernel_size=2, stride=2)
        pool1_end_h = self.pool2d_output_size(conv1_end_h, kernel_size=2, stride=2)

        conv2_end_w = self.conv2d_output_size(pool1_end_w, kernel_size=5)
        conv2_end_h = self.conv2d_output_size(pool1_end_h, kernel_size=5)

        pool2_end_w = self.pool2d_output_size(conv2_end_w, kernel_size=2, stride=2)
        pool2_end_h = self.pool2d_output_size(conv2_end_h, kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(self.num_input_channels, hidden_channels_1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(hidden_channels_1, hidden_channels_2, 5)
        self.fc1 = nn.Linear(hidden_channels_2 * pool2_end_h * pool2_end_w, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.config["model"]["HyraxCNN"]["output_classes"])

    def conv2d_output_size(self, input_size, kernel_size, padding=0, stride=1, dilation=1) -> int:
        # From https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        numerator = input_size + 2 * padding - dilation * (kernel_size - 1) - 1
        return int((numerator / stride) + 1)

    def pool2d_output_size(self, input_size, kernel_size, stride, padding=0, dilation=1) -> int:
        # From https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        numerator = input_size + 2 * padding - dilation * (kernel_size - 1) - 1
        return int((numerator / stride) + 1)

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
