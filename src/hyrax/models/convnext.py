# ruff: noqa: D101, D102
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa N812
import torchvision.transforms as T
import numpy as np
from torchvision.transforms.v2 import CenterCrop
# from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models.convnext import ConvNeXt, CNBlockConfig
from sklearn.datasets import make_blobs

# extra long import here to address a circular import issue
from hyrax.models.model_registry import hyrax_model

logger = logging.getLogger(__name__)

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


class KMeansLoss(nn.Module):
    """
    GPU-based loss computation
    """
    def __init__(self, n_clusters, embedding_dim):
        """
        Parameters
        ----------
        n_clusters : int
            Number of clusters
        embedding_dim : int
            Dimension of the embeddings
        """
        super().__init__()

        self.centroids = nn.Parameter(
            torch.randn(n_clusters, embedding_dim)
        )

    def forward(self, embeddings):
        # embeddings: (batch_size, embedding_dim)

        # compute squared distances
        distances = torch.cdist(embeddings, self.centroids) ** 2

        # assign closest centroid
        min_distances, assignments = distances.min(dim=1)

        # loss = mean distance to assigned centroid
        loss = min_distances.mean()

        return loss
    

@hyrax_model
class HyraxConvNeXt(nn.Module):
    """
    ConvNeXt model based on Liu+2020. Implemented withing `torchvision` standard library.

    Implemented as an autoencoder
    """
    def __init__(self, config, data_sample=None):
        super().__init__()
        self.config = config

        # Number of classes
        self.num_classes = config["model"]["HyraxConvNeXt"]["num_classes"]

        # Extract the shape from an image
        shape = data_sample[0].shape

        self.num_input_channels, self.image_width, self.image_height = shape

        # Create a CNBlock for the model
        # (input channels, output channels, num layers)
        # Reverse if used for the decoder
        # self.block_setting = [
        #     CNBlockConfig(96, 192, 1),
        #     CNBlockConfig(192, 384, 1),
        #     CNBlockConfig(384, 768, 1),
        #     CNBlockConfig(768, 768, 1),
        # ]

        self.block_setting = [
            CNBlockConfig(96, 128, 1),
            CNBlockConfig(128, 256, 1),
            CNBlockConfig(256, 128, 1),
        ]

        # ConvNeXt is used as the backbone
        self.backbone = ConvNeXt(
            block_setting=self.block_setting,
            stochastic_depth_prob=0.1,
            num_classes=self.num_classes,
        )

        # Override criterion
        self.criterion = KMeansLoss(50, self.num_classes)


    def forward(self, x):
        # TODO: How should I handle the dimensions before feeding the data to the model
        return self.backbone(x)
    
    def train_batch(self, batch):
        # Create embeddings
        embeddings = self.forward(batch[0])

        criterion = self.criterion

        loss = criterion(embeddings)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def validate_batch(self, batch):
        # Create embeddings
        embeddings = self.forward(batch[0])

        criterion = self.criterion

        loss = criterion(embeddings)

        return {"loss": loss.item()}


    def test_batch(self, batch):
        raise NotImplementedError()

    def infer_batch(self, batch):
        raise NotImplementedError()
    




