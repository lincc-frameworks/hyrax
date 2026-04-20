# ruff: noqa: D101, D102
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa N812
import torchvision.transforms as T  # noqa N812
from torchvision.transforms.v2 import CenterCrop

# extra long import here to address a circular import issue
from hyrax.models.model_registry import hyrax_model

logger = logging.getLogger(__name__)


class ArcsinhActivation(nn.Module):
    """Helper module for HyraxAutoencoderV2 to use the arcsinh function"""

    def forward(self, x):
        return torch.arcsinh(x)


class APCTTransform(nn.Module):
    """
    Perform Adaptive Polar Coordinate Transformation (APCT), mostly based on Fang et al 2023

    To be used with `torchvision.transforms`
    """

    def __init__(self, ref_channel=3):
        super().__init__()

        # The index of channels used as pivot for all channels
        # Default at 3 (LSST's i-channel)
        self.ref_channel = ref_channel

    # TODO: Check if this argument signature aligns with that of SimCLR
    def forward(self, imgs):
        # Transform the image
        new_imgs = self.transform_apct_nchannels(imgs, self.ref_channel)

        # No change in label
        return new_imgs

    def transform_apct_nchannels(
        self, image: torch.Tensor, ref_channel: int = 2, output_size: tuple[int, int] = (106, 125)
    ):
        """
        Multiple-channel implementation of APCT in Fang et al. 2023

        Parameters
        ----------
        image : tensor
            tensor image to be transformed
        ref_channel : int
            The index of the channels to be used as a reference. Default to i-channel.
        output_size : tuple[int, int]
            The output dimensions of the transformed image.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # -> (1, C, H, W)

        batches, channels, height, width = image.shape

        ref_img = image[:, ref_channel, :, :]

        # Sort the flux in all pixels, in an ascending order
        argsorted_indices = torch.argsort(ref_img.flatten())

        # Locate the maximum
        max_flat_index = argsorted_indices[-1]
        max_indices = torch.unravel_index(max_flat_index, (height, width))

        # Locate the minimum
        min_flat_index = argsorted_indices[1]
        min_indices = torch.unravel_index(min_flat_index, (height, width))

        ## Transform to polar coordinate
        # Use the maximum as the origin and the reference angle toward the minimum
        ref_angle = np.atan2(
            min_indices[1].cpu() - max_indices[1].cpu(), min_indices[0].cpu() - max_indices[0].cpu()
        )

        # Find the center of the image
        # The center of the image is the indices of the maximum pixel + 1
        cx, cy = max_indices[0].cpu() + 1, max_indices[1].cpu() + 1
        radius = np.sqrt(cx**2 + cy**2)

        # Build sampling grid in polar space
        # Offset theta space by the angle from +x
        theta = torch.linspace(0, 2 * torch.pi, output_size[1], device=image.device) - np.round(
            ref_angle / 0.05
        )
        r = torch.linspace(0, radius, output_size[0], device=image.device)

        # Meshgrid: (n_angles, n_radii)
        theta_grid, r_grid = torch.meshgrid(theta, r, indexing="ij")

        # Convert polar -> Cartesian pixel coords
        x = r_grid * torch.cos(theta_grid) + cx
        y = r_grid * torch.sin(theta_grid) + cy

        # Normalize to [-1, 1] as required by grid_sample
        x_norm = (x / (width - 1)) * 2 - 1
        y_norm = (y / (height - 1)) * 2 - 1

        grid = torch.stack([x_norm, y_norm], dim=-1)  # (n_angles, n_radii, 2)

        grid = grid.unsqueeze(0).expand(batches, -1, -1, -1)  # (B, n_angles, n_radii, 2)

        warped = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

        mirroreds = torch.concatenate([torch.flip(warped, dims=(3,)), warped], dim=3)

        rotated_mirroreds = torch.rot90(mirroreds, k=-1, dims=(2, 3))
        return rotated_mirroreds  # (B, C, 2 * n_radii, n_angles)


@hyrax_model
class HyraxAutoencoderV2APCT(nn.Module):
    """
    This is tweaked version of HyraxAutoencoder and is designed to work with a wide range of imaging datasets.

    V2 improvements:
    - Configurable final layer activation
    - Uses criterion and optimizer from config variables
    """

    def __init__(self, config, data_sample=None):
        super().__init__()
        self.config = config

        aug = T.Compose(
            [
                APCTTransform(ref_channel=2),
            ]
        )
        print(data_sample.shape, type(data_sample))

        shape = aug(torch.from_numpy(data_sample)).shape
        print(shape)
        logger.debug(f"Found shape: {shape} in data sample, using this to initialize model.")

        _, self.num_input_channels, self.image_width, self.image_height = shape

        self.c_hid = self.config["model"]["HyraxAutoencoderV2APCT"]["base_channel_size"]
        self.latent_dim = self.config["model"]["HyraxAutoencoderV2APCT"]["latent_dim"]

        # Calculate how much our convolutional layers will affect the size of final convolution
        # Formula evaluated from: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        #
        # If the number of layers are changed this will need to be rewritten.
        self.conv_end_w = self.conv2d_multi_layer(self.image_width, 3, kernel_size=3, padding=1, stride=2)
        self.conv_end_h = self.conv2d_multi_layer(self.image_height, 3, kernel_size=3, padding=1, stride=2)

        self._init_encoder()
        self._init_decoder()

        # Configurable band reduction strategy
        self.band_reduction = self.config["criterion"]["band_loss_reduction"]

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
        final_layer_value = self.config["model"]["HyraxAutoencoderV2APCT"]["final_layer"]
        final_layer = final_layer_value if final_layer_value else "tanh"
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

    def train_batch(self, batch):
        """This function contains the logic for a single training step. i.e. the
        contents of the inner loop of a ML training process.

        Parameters
        ----------
        batch : tuple
            A tuple containing the input data for the current batch, possibly
            with labels that are ignored.

        Returns
        -------
        Current loss value : dict
            Dictionary containing the loss value for the current batch.
        """
        aug = T.Compose(
            [
                APCTTransform(ref_channel=2),
            ]
        )

        x = aug(batch[0])

        z = self._eval_encoder(x)
        x_hat = self._eval_decoder(z)

        # The loss averaging strategy here is different from v1 which averages
        # over only the batch dimension. Here we always average over both batch
        # and spaital dimensions; so as the loss-value is not impacted by image size.
        if self.band_reduction == "sum":
            # Sum across bands, mean over spatial dims and batch
            # More channels will result in larger loss values
            # but MIGHT result in better popping out of bad reconstruction
            # in a single band/channel
            criterion_cls = type(self.criterion)
            loss = criterion_cls(reduction="none")(x_hat, x)
            loss = loss.sum(dim=1).mean()
        elif self.band_reduction == "mean":
            # Default: Mean over all dimensions (batch,channel,spatial)
            loss = self.criterion(x_hat, x)
        else:
            raise ValueError(
                f"band_loss_reduction:{self.band_reduction} not supported by HyraxAutoencoderV2.\
                               Current supported options are sum and mean (default)"
            )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return {"loss": loss.item()}

    def validate_batch(self, batch):
        """This function contains the logic for a single validation step that will
        process a single batch of data. i.e. the contents of the inner loop of a
        ML validation process.

        Parameters
        ----------
        batch : tuple
            A tuple containing the input data for the current batch, possibly
            with labels that are ignored.

        Returns
        -------
        Current loss value : dict
            Dictionary containing the loss value for the current batch.
        """
        aug = T.Compose(
            [
                APCTTransform(ref_channel=2),
            ]
        )

        x = aug(batch[0])
        z = self._eval_encoder(x)
        x_hat = self._eval_decoder(z)

        if self.band_reduction == "sum":
            criterion_cls = type(self.criterion)
            loss = criterion_cls(reduction="none")(x_hat, x)
            loss = loss.sum(dim=1).mean()
        elif self.band_reduction == "mean":
            loss = self.criterion(x_hat, x)
        else:
            raise ValueError(
                f"band_loss_reduction:{self.band_reduction} not supported by HyraxAutoencoderV2.\
                               Current supported options are sum and mean (default)"
            )

        return {"loss": loss.item()}

    def test_batch(self, batch):
        """This function contains the logic for a single testing step that will
        process a single batch of data. i.e. the contents of the inner loop of a
        ML testing process. In this case, it is identical to `validate_batch`.

        Parameters
        ----------
        batch : tuple
            A tuple containing the input data for the current batch, possibly
            with labels that are ignored.

        Returns
        -------
        Current loss value : dict
            Dictionary containing the loss value for the current batch.
        """
        x = batch
        z = self._eval_encoder(x)
        x_hat = self._eval_decoder(z)

        if self.band_reduction == "sum":
            criterion_cls = type(self.criterion)
            loss = criterion_cls(reduction="none")(x_hat, x)
            loss = loss.sum(dim=1).mean()
        elif self.band_reduction == "mean":
            loss = self.criterion(x_hat, x)
        else:
            raise ValueError(
                f"band_loss_reduction:{self.band_reduction} not supported by HyraxAutoencoderV2.\
                               Current supported options are sum and mean (default)"
            )

        return {"loss": loss.item()}

    def infer_batch(self, batch):
        """This function contains the logic for a single inference step. i.e. the
        contents of the inner loop of a ML inference process.

        Parameters
        ----------
        batch : tuple
            A tuple containing the input data for the current batch, possibly
            with labels that are ignored.

        Returns
        -------
        Reconstructed outputs : torch.Tensor
            The reconstructed outputs from the autoencoder.
        """
        aug = T.Compose(
            [
                APCTTransform(ref_channel=2),
            ]
        )

        x = aug(batch)
        return self.forward(x)

    @staticmethod
    def prepare_inputs(data_dict) -> tuple:
        """This function converts structured data to the input tensor we need to run

        Parameters
        ----------
        data_dict : dict
            The dictionary returned from our data source
        """
        if "data" not in data_dict:
            raise RuntimeError("Unable to find `data` key in data_dict")

        data_dict = data_dict["data"]
        if "image" in data_dict:
            return data_dict["image"]
        else:
            raise RuntimeError("Data dict did not contain image key.")
