# ruff: noqa: D101, D102


import torch.nn as nn

# extra long import here to address a circular import issue
from hyrax.models.model_registry import hyrax_model


@hyrax_model
class HSCAutoencoder(nn.Module):  # These shapes work with [3,258,258] inputs
    """
    This autoencoder is designed to work with datasets that are prepared with Hyrax's HSC Data Set class.
    """

    def __init__(self, config, shape):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=4, output_padding=1),
            nn.Sigmoid(),  # Output pixel values between 0 and 1
        )

        self.config = config

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_step(self, batch):
        """
        This function contains the logic for a single training step. i.e. the
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

        data = batch[0]
        self.optimizer.zero_grad()

        decoded = self.forward(data)
        loss = self.criterion(decoded, data)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}
