# ruff: noqa: D101, D102


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa N812
import torchvision.models as models
import torchvision.transforms as T  # noqa N812
from scipy.ndimage import shift
from skimage.transform import warp_polar

from hyrax.models.model_registry import hyrax_model


class APCTTransform(torch.nn.Module):
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

    def transform_apct_nchannels(self, imgs, ref_channel):
        """
        Multiple-channel implementation of APCT in Fang et al. 2023

        This routine assumes Rubin's 6 channels data by default

        Parameters
        ----------
        imgs : array_like
            An array of images of length N that need to be transform. Should have N channels.
        ref_channel : int, default=3
            The index of the channels to be used as a reference. Default to i-channel.

        Returns
        -------
        unrolled_img : ndarray
            An array of length N of polar-transformed images.
        """
        # Get the shape of the image
        shape = imgs.shape[1:3]

        # This routine uses the reference channel as an anchor and
        # applies the same transformation to the other channels
        ref_img = imgs[ref_channel, :, :]

        # Sort the flux in all pixels, in an ascending order
        argsorted_indices = np.argsort(ref_img.flatten())

        # Locate the maximum
        max_flat_index = argsorted_indices[-1]
        max_indices = np.unravel_index(max_flat_index, shape)

        # Locate the minimum
        min_flat_index = argsorted_indices[1]
        min_indices = np.unravel_index(min_flat_index, shape)

        ## Transform to polar coordinate
        # Use the maximum as the origin and the reference angle toward the minimum
        ref_angle = np.atan2(min_indices[1] - max_indices[1], min_indices[0] - max_indices[0])

        # Warp the image and transform to polar coordinates
        # Apply to all images with the channel axis being 0
        unrolled_imgs = warp_polar(
            imgs,
            center=(max_indices[0], max_indices[1]),
            output_shape=(125, shape[0] // 2),
            radius=shape[0] // 2 * np.sqrt(2),
            scaling="linear",
            channel_axis=0,
        )

        # Shift the unrolled image to align with the reference vector
        # Each x-pixel correspond to 0.05 radian increment. Wrap the images around
        def unroll(unrolled_img):
            return shift(unrolled_img, (-np.round(ref_angle / 0.05), 0), mode="wrap")

        unrolled_imgs_shifted = np.array([unroll(im) for im in unrolled_imgs])

        # Mirror the image along y axis
        mirroreds = np.concatenate([np.flip(unrolled_imgs_shifted, axis=2), unrolled_imgs_shifted], axis=2)

        # The end result is rotated so that it's in landscape orientation with (0, 0) on the left side
        return np.rot90(mirroreds, k=-1, axes=(1, 2))


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss. Based on Chen, 2020"""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        """Forward function of NTXentLoss. Based on Chen, 2020.
        Loss is calculated from representations from two augmented views of the same batch.
        """
        batch_size = z_i.shape[0]
        device = z_i.device

        # Normalize the matrix and concat
        z_i = F.normalize(z_i, dim=1)  # Shape: (N, D)
        z_j = F.normalize(z_j, dim=1)  # Shape: (N, D)
        z = torch.cat([z_i, z_j], dim=0)  # Shape: (2N, D)

        # Cosine similarity
        sim_matrix = torch.matmul(z, z.T)  # Shape: (2N, 2N)

        # Remove self-similarity by masking the diagonal
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
        sim_matrix = sim_matrix.masked_fill(mask, -float("inf"))

        # Apply temperature scaling
        sim_matrix /= self.temperature

        # Construct positive pair indices: Each example i has its positive pair at index i + N or i - N
        positive_indices = (torch.arange(0, 2 * batch_size, device=device) + batch_size) % (2 * batch_size)

        # Compute cross-entropy loss (it's mathematically equivalent)
        loss = self.criterion(sim_matrix, positive_indices)
        loss /= 2 * batch_size

        return loss


class PositiveRescale:
    """Transformation Class specifically for ColorJitter to prevent wrong domain during the augmentation"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        x = (x + 1) / 2  # to [0, 1]
        x = self.transform(x)
        return x * 2 - 1  # back to (-1, 1)


@hyrax_model
class SimCLR(nn.Module):
    """SimCLR model. Implementation based on Chen, 2020"""

    def __init__(self, config, data_sample=None):
        super().__init__()
        self.config = config

        if data_sample is None:
            raise ValueError("A `data_sample` must be provided for dynamic sizing.")

        self.shape = data_sample[0].shape
        proj_dim = config["model"]["SimCLR"]["projection_dimension"]
        temperature = config["model"]["SimCLR"]["temperature"]

        backbone = models.resnet18(pretrained=False)
        # final_layer = self.config["model"].get("final_layer", "tanh")
        # if final_layer == "sigmoid":
        #     self.final_activation = nn.Sigmoid()
        # elif final_layer == "tanh":
        #     self.final_activation = nn.Tanh()
        # elif final_layer == "arcsinh":
        #     self.final_activation = ArcsinhActivation()
        # elif final_layer == "identity":
        #     self.final_activation = nn.Identity()
        # else:
        #     self.final_activation = nn.Tanh()
        backbone.fc = self.final_activation
        self.backbone = backbone

        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

        # TODO: Make sure to revisit this and properly implement custom criterion
        self.the_criterion = NTXentLoss(temperature)

    def forward(self, x):
        feats = self.backbone(x)
        return self.projection_head(feats)

    def train_batch(self, x):
        # Extract the tensors from the single-element tuple
        x = x[0]

        aug = T.Compose(
            [
                T.RandomResizedCrop(size=x.shape[-1]),
                T.RandomHorizontalFlip(self.config["model"]["SimCLR"]["horizontal_flip_probability"]),
                T.RandomApply(
                    [PositiveRescale(T.ColorJitter(*self.config["model"]["SimCLR"]["color_jitter_params"]))],
                    p=self.config["model"]["SimCLR"]["color_jitter_probability"],
                ),
                T.RandomGrayscale(p=self.config["model"]["SimCLR"]["grayscale_probability"]),
                T.GaussianBlur(
                    kernel_size=self.config["model"]["SimCLR"]["gaussian_blur_kernel_size"],
                    sigma=self.config["model"]["SimCLR"]["gaussian_blur_sigma_range"],
                ),
            ]
        )

        x1 = torch.stack([aug(img) for img in x])
        x2 = torch.stack([aug(img) for img in x])

        z1 = self.forward(x1)
        z2 = self.forward(x2)

        loss = self.criterion(z1, z2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def validate_batch(self, x):
        # Extract the tensors from the single-element tuple
        x = x[0]

        aug = T.Compose(
            [
                T.RandomResizedCrop(size=x.shape[-1]),
                T.RandomHorizontalFlip(self.config["model"]["SimCLR"]["horizontal_flip_probability"]),
                T.RandomApply(
                    [PositiveRescale(T.ColorJitter(*self.config["model"]["SimCLR"]["color_jitter_params"]))],
                    p=self.config["model"]["SimCLR"]["color_jitter_probability"],
                ),
                T.RandomGrayscale(p=self.config["model"]["SimCLR"]["grayscale_probability"]),
                T.GaussianBlur(
                    kernel_size=self.config["model"]["SimCLR"]["gaussian_blur_kernel_size"],
                    sigma=self.config["model"]["SimCLR"]["gaussian_blur_sigma_range"],
                ),
            ]
        )

        x1 = torch.stack([aug(img) for img in x])
        x2 = torch.stack([aug(img) for img in x])

        z1 = self.forward(x1)
        z2 = self.forward(x2)

        loss = self.criterion(z1, z2)
        return {"loss": loss.item()}

    def test_batch(self, x):
        return self.validate_batch(x)

    def infer_batch(self, x):
        """Function to run inference on a batch of data.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, projection_dimension).
        """
        return self.forward(x)
