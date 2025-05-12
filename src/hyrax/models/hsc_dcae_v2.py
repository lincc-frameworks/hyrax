# ruff: noqa: D101, D102

# This is a more felxible version of hsc_dcae.py that should
# work with a variety of image sizes.

import torch
import torch.nn as nn
import torch.nn.functional as f

# extra long import here to address a circular import issue
from hyrax.models.model_registry import hyrax_model


class ArcsinhActivation(nn.Module):
    """Helper module for HSCDAEv2 to use the arcsinh function"""

    def forward(self, x):
        return torch.arcsinh(x)


@hyrax_model
class HSCDCAEv2(nn.Module):
    """
    This autoencoder is designed to work with datasets that are prepared with Hyrax's HSC Data Set class.
    """

    def __init__(self, config, shape):
        super().__init__()

        # Store input shape for dynamic sizing
        self.input_shape = shape

        # Extract number of channels from input shape (assuming NCHW format)
        in_channels = shape[1] if len(shape) > 3 else shape[0]

        # Encoder
        self.encoder1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.encoder4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Decoder - using normal Conv2d with upsampling instead of ConvTranspose2d
        # This approach is more flexible for different image sizes
        self.decoder4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.decoder1 = nn.Conv2d(32, in_channels, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU()

        # Configure final activation
        final_layer = config["model"].get("HSCDCAE_final_layer", "identity")
        if final_layer == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif final_layer == "tanh":
            self.final_activation = nn.Tanh()
        elif final_layer == "arcsinh":
            self.final_activation = ArcsinhActivation()
        else:
            self.final_activation = nn.Identity()

        self.config = config

    def encode(self, x):
        # Encoder with skip connections
        x1 = self.activation(self.encoder1(x))
        p1 = self.pool(x1)

        x2 = self.activation(self.encoder2(p1))
        p2 = self.pool(x2)

        x3 = self.activation(self.encoder3(p2))
        p3 = self.pool(x3)

        x4 = self.activation(self.encoder4(p3))

        return x4, [x3, x2, x1]

    def decode(self, x, skip_connections):
        # Decoder with skip connections and dynamic upsampling
        x = f.interpolate(x, size=skip_connections[0].shape[2:], mode="bilinear", align_corners=False)
        x = self.activation(self.decoder4(x) + skip_connections[0])

        x = f.interpolate(x, size=skip_connections[1].shape[2:], mode="bilinear", align_corners=False)
        x = self.activation(self.decoder3(x) + skip_connections[1])

        x = f.interpolate(x, size=skip_connections[2].shape[2:], mode="bilinear", align_corners=False)
        x = self.activation(self.decoder2(x) + skip_connections[2])

        # Final interpolation to input size
        if hasattr(self, "original_size"):
            x = f.interpolate(x, size=self.original_size, mode="bilinear", align_corners=False)

        x = self.final_activation(self.decoder1(x))

        return x

    def forward(self, x):
        # Dropping labels if present
        x = x[0] if isinstance(x, tuple) else x

        # Store original spatial dimensions for decoding
        self.original_size = x.shape[2:]

        # Encode
        encoded, skip_connections = self.encode(x)

        return encoded

    def train_step(self, batch):
        """This function contains the logic for a single training step.

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

        # Store original spatial dimensions for decoding
        self.original_size = data.shape[2:]

        # Encode
        encoded, skip_connections = self.encode(data)

        # Decode
        decoded = self.decode(encoded, skip_connections)

        # Compute loss
        loss = self.criterion(decoded, data)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}
