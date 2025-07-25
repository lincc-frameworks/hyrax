# ruff: noqa: D101, D102


import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa N812
from torch import Tensor
from torchvision.transforms.v2 import CenterCrop

# extra long import here to address a circular import issue
from hyrax.models.model_registry import hyrax_model


class ArcsinhActivation(nn.Module):
    """Helper module for HyraxAutoencoderV2 to use the arcsinh function"""

    def forward(self, x):
        return torch.arcsinh(x)


@hyrax_model
class HyraxAutoencoderV2(nn.Module):
    """
    This is tweaked version of HyraxAutoencoder and is designed to work with a wide range of imaging datasets.

    V2 improvements:
    - Configurable final layer activation
    - Uses criterion and optimizer from config variables
    """

    def __init__(self, config, shape=(5, 250, 250)):
        super().__init__()
        self.config = config

        # TODO config-ize or get from data loader somehow
        self.num_input_channels, self.image_width, self.image_height = shape

        self.c_hid = self.config["model"]["base_channel_size"]
        self.latent_dim = self.config["model"]["latent_dim"]

        # Calculate how much our convolutional layers will affect the size of final convolution
        # Formula evaluated from: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        #
        # If the number of layers are changed this will need to be rewritten.
        self.conv_end_w = self.conv2d_multi_layer(self.image_width, 3, kernel_size=3, padding=1, stride=2)
        self.conv_end_h = self.conv2d_multi_layer(self.image_height, 3, kernel_size=3, padding=1, stride=2)

        self._init_encoder()
        self._init_decoder()

    def conv2d_multi_layer(self, input_size, num_applications, **kwargs) -> int:
        for _ in range(num_applications):
            input_size = self.conv2d_output_size(input_size, **kwargs)

        return int(input_size)

    def conv2d_output_size(self, input_size, kernel_size, padding=0, stride=1, dilation=1) -> int:
        # From https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        numerator = input_size + 2 * padding - dilation * (kernel_size - 1) - 1
        return int((numerator / stride) + 1)

    def _init_encoder(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.c_hid, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(self.c_hid, self.c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.c_hid, 2 * self.c_hid, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(2 * self.c_hid, 2 * self.c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * self.c_hid, 2 * self.c_hid, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * self.conv_end_h * self.conv_end_w * self.c_hid, self.latent_dim),
        )

    def _eval_encoder(self, x):
        return self.encoder(x)

    def _init_decoder(self):
        self.dec_linear = nn.Sequential(
            nn.Linear(self.latent_dim, 2 * self.conv_end_h * self.conv_end_w * self.c_hid), nn.GELU()
        )

        # Configure final activation
        # Should be set to the same value as ["dataset"]["transform"] in most cases
        final_layer = self.config["model"].get("final_layer", "tanh")
        if final_layer == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif final_layer == "tanh":
            self.final_activation = nn.Tanh()
        elif final_layer == "arcsinh":
            self.final_activation = ArcsinhActivation()
        elif final_layer == "identity":
            self.final_activation = nn.Identity()
        else:
            self.final_activation = nn.Tanh()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                2 * self.c_hid, 2 * self.c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 4x4 => 8x8
            nn.GELU(),
            nn.Conv2d(2 * self.c_hid, 2 * self.c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(
                2 * self.c_hid, self.c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 8x8 => 16x16
            nn.GELU(),
            nn.Conv2d(self.c_hid, self.c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.c_hid, self.num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 16x16 => 32x32
            self.final_activation,
        )

    def _eval_decoder(self, x):
        x = self.dec_linear(x)
        x = x.reshape(x.shape[0], -1, self.conv_end_h, self.conv_end_w)
        x = self.decoder(x)
        x = CenterCrop(size=(self.image_width, self.image_height))(x)
        return x

    def forward(self, batch):
        return self._eval_encoder(batch)

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
        x = batch
        z = self._eval_encoder(x)
        x_hat = self._eval_decoder(z)

        # Configurable band reduction strategy
        band_reduction = self.config["criterion"].get("band_loss_reduction", "mean")

        # The loss averaging strategy here is different from v1 which averages
        # over only the batch dimension. Here we always average over both batch
        # and spaital dimensions; so as the loss-value is not impacted by image size.
        if band_reduction == "sum":
            # Sum across bands, mean over spatial dims and batch
            # More channels will result in larger loss values
            # but MIGHT result in better popping out of bad reconstruction
            # in a single band/channel
            criterion_cls = type(self.criterion)
            loss = criterion_cls(reduction="none")(x_hat, x)
            loss = loss.sum(dim=1).mean()
        elif band_reduction == "mean":
            # Default: Mean over all dimensions (batch,channel,spatial)
            loss = self.criterion(x_hat, x)
        else:
            raise ValueError(
                f"band_loss_reduction:{band_reduction} not supported by HyraxAutoencoderV2.\
                               Current supported options are sum and mean (default)"
            )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return {"loss": loss.item()}

    @staticmethod
    def to_tensor(data_dict) -> tuple[Tensor]:
        """This function converts structured data to the input tensor we need to run

        Parameters
        ----------
        data_dict : dict
            The dictionary returned from our data source
        """
        if "image" in data_dict:
            return data_dict["image"]
        else:
            raise RuntimeError("Data dict did not contain image key.")
