# ruff: noqa: D101, D102

# This is a more flexible version of hsc_dcae.py that should
# work with a variety of image sizes and includes a true latent bottleneck
# for better anomaly detection capabilities.

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

# extra long import here to address a circular import issue
from hyrax.models.model_registry import hyrax_model


class ArcsinhActivation(nn.Module):
    """Helper module for ImageDCAE to use the arcsinh function"""

    def forward(self, x):
        return torch.arcsinh(x)


@hyrax_model
class ImageDCAE(nn.Module):
    """
    This is an autoencoder with skipconnections that should work with
    arbitarily sized images with arbitrary number of channels.
    """

    def __init__(self, config, shape):
        super().__init__()

        # Store input shape for dynamic sizing
        self.input_shape = shape
        self.config = config

        # Extract dimensions from input shape
        if len(shape) == 4:  # Batch dimension included
            self.num_input_channels = shape[1]
            self.image_height, self.image_width = shape[2], shape[3]
        else:  # No batch dimension
            self.num_input_channels = shape[0]
            self.image_height, self.image_width = shape[1], shape[2]

        # Get latent dimension from config (similar to HyraxAutoencoder)
        self.latent_dim = config["model"].get("latent_dim", 512)
        self.base_channel_size = config["model"].get("base_channel_size", 32)

        # Calculate the size after convolutional layers for the linear bottleneck
        self.conv_output_size = self._calculate_conv_output_size()

        in_channels = self.num_input_channels

        # Encoder - using configurable base channel size
        self.encoder1 = nn.Conv2d(in_channels, self.base_channel_size, kernel_size=3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(
            self.base_channel_size, self.base_channel_size * 2, kernel_size=3, stride=1, padding=1
        )
        self.encoder3 = nn.Conv2d(
            self.base_channel_size * 2, self.base_channel_size * 4, kernel_size=3, stride=1, padding=1
        )
        self.encoder4 = nn.Conv2d(
            self.base_channel_size * 4, self.base_channel_size * 8, kernel_size=3, stride=1, padding=1
        )

        self.pool = nn.MaxPool2d(2, 2)

        # Latent bottleneck - similar to HyraxAutoencoder
        self.latent_encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(self.conv_output_size, self.latent_dim), nn.GELU()
        )

        self.latent_decoder = nn.Sequential(nn.Linear(self.latent_dim, self.conv_output_size), nn.GELU())

        # Decoder - using normal Conv2d with upsampling instead of ConvTranspose2d
        # This approach is more flexible for different image sizes
        self.decoder4 = nn.Conv2d(
            self.base_channel_size * 8, self.base_channel_size * 4, kernel_size=3, stride=1, padding=1
        )
        self.decoder3 = nn.Conv2d(
            self.base_channel_size * 4, self.base_channel_size * 2, kernel_size=3, stride=1, padding=1
        )
        self.decoder2 = nn.Conv2d(
            self.base_channel_size * 2, self.base_channel_size, kernel_size=3, stride=1, padding=1
        )
        self.decoder1 = nn.Conv2d(self.base_channel_size, in_channels, kernel_size=3, stride=1, padding=1)

        self.activation = nn.GELU()  # Better gradients than ReLU

        # Configure final activation
        final_layer = config["model"].get("final_layer", "identity")
        if final_layer == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif final_layer == "tanh":
            self.final_activation = nn.Tanh()
        elif final_layer == "arcsinh":
            self.final_activation = ArcsinhActivation()
        else:
            self.final_activation = nn.Identity()

    def _calculate_conv_output_size(self):
        """Calculate the output size after all convolutional layers for the linear bottleneck."""
        # Simulate the forward pass through conv layers to get the size
        h, w = self.image_height, self.image_width

        # After 3 pooling operations (each divides by 2)
        h = h // 8
        w = w // 8

        # Final feature map size: (base_channel_size * 8) * h * w
        return (self.base_channel_size * 8) * h * w

    def encode(self, x):
        """Encode input to latent space with skip connections."""
        # Encoder with skip connections
        x1 = self.activation(self.encoder1(x))
        p1 = self.pool(x1)

        x2 = self.activation(self.encoder2(p1))
        p2 = self.pool(x2)

        x3 = self.activation(self.encoder3(p2))
        p3 = self.pool(x3)

        x4 = self.activation(self.encoder4(p3))

        # Store the spatial dimensions for reconstruction
        self.encoded_spatial_shape = x4.shape

        # Pass through latent bottleneck
        latent = self.latent_encoder(x4)

        return latent, [x3, x2, x1], x4.shape

    def decode(self, latent, skip_connections, encoded_shape):
        """Decode from latent space to image with skip connections."""
        # Reconstruct from latent space
        x = self.latent_decoder(latent)

        # Reshape back to convolutional feature map
        x = x.reshape(encoded_shape)

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
        """Forward pass - returns latent representation for anomaly detection."""
        # Store original spatial dimensions for decoding
        self.original_size = x.shape[2:]

        # Encode to latent space
        latent, skip_connections, encoded_shape = self.encode(x)

        return latent

    def reconstruct(self, x):
        """Full reconstruction for evaluation and anomaly detection."""
        # Dropping labels if present
        x = x[0] if isinstance(x, tuple) else x

        # Store original spatial dimensions for decoding
        self.original_size = x.shape[2:]

        # Encode to latent space
        latent, skip_connections, encoded_shape = self.encode(x)

        # Decode back to image
        reconstructed = self.decode(latent, skip_connections, encoded_shape)

        return reconstructed

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
        data = batch
        self.optimizer.zero_grad()

        # Store original spatial dimensions for decoding
        self.original_size = data.shape[2:]

        # Encode to latent space
        latent, skip_connections, encoded_shape = self.encode(data)

        # Decode back to image
        decoded = self.decode(latent, skip_connections, encoded_shape)

        # Compute loss
        loss = self.criterion(decoded, data)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def _optimizer(self):
        """Default optimizer configuration."""
        return optim.Adam(self.parameters(), lr=1e-3)

    @staticmethod
    def to_tensor(data_dict):
        """Convert structured data to tensor format."""
        if "image" in data_dict:
            return data_dict["image"]
        else:
            raise RuntimeError("Data dict did not contain image key.")
