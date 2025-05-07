# ruff: noqa: D101, D102

# This autoencoder is designed to work with datasets
# that are prepared with Hyrax's HSC Data Set class.


import torch
import torch.nn as nn

# extra long import here to address a circular import issue
from hyrax.models.model_registry import hyrax_model


class ArcsinhActivation(nn.Module):
    """Helper module for HSCDAE to use the arcsinh function"""

    def forward(self, x):
        return torch.arcsinh(x)


@hyrax_model
class HSCDCAE(nn.Module):
    """
    This autoencoder is designed to work with datasets that are prepared with Hyrax's HSC Data Set class.
    """

    def __init__(self, config, shape):
        super().__init__()

        # The current network works with images of size [3,150,150]
        # You will need to updat padding, stride, etc. for imags
        # of other sizes

        # Encoder
        self.encoder1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.encoder4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.decoder4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0, output_padding=0)
        self.decoder3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0, output_padding=0)
        self.decoder2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0, output_padding=0)
        self.decoder1 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1, output_padding=0)

        self.activation = nn.ReLU()

        final_layer = config["model"]["HSCDCAE_final_layer"]
        if final_layer == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif final_layer == "tanh":
            self.final_activation = nn.Tanh()
        elif final_layer == "arcsinh":
            self.final_activation = ArcsinhActivation()
        else:
            self.final_activation = nn.Identity()

        self.config = config

    def forward(self, x):
        # Dropping labels if present
        x = x[0] if isinstance(x, tuple) else x

        # Encoder with skip connections
        x1 = self.activation(self.encoder1(x))
        x2 = self.activation(self.encoder2(self.pool(x1)))
        x3 = self.activation(self.encoder3(self.pool(x2)))
        x4 = self.activation(self.encoder4(self.pool(x3)))

        return x4

    def train_step(self, batch):
        """This function contains the logic for a single training step. i.e. the
        contents of the inner loop of a ML training process.

        Parameters
        ----------
        batch : tuple
            A tuple containing the two values the loss function

        Returns
        -------
        Current loss value : dict
            Dictionary containing the loss value for the current batch.
        """

        # Dropping labels if present
        data = batch[0] if isinstance(batch, tuple) else batch
        self.optimizer.zero_grad()

        # Encoder with skip connections
        x1 = self.activation(self.encoder1(data))
        x2 = self.activation(self.encoder2(self.pool(x1)))
        x3 = self.activation(self.encoder3(self.pool(x2)))
        x4 = self.activation(self.encoder4(self.pool(x3)))

        # Decoder with skip connections
        x = self.activation(self.decoder4(x4) + x3)
        x = self.activation(self.decoder3(x) + x2)
        x = self.activation(self.decoder2(x) + x1)
        decoded = self.final_activation(self.decoder1(x))

        loss = self.criterion(decoded, data)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}
