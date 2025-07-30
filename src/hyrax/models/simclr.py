# ruff: noqa: D101, D102


import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa N812
import torchvision.models as models
import torchvision.transforms as T  # noqa N812

from hyrax.models.model_registry import hyrax_model


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

    def __init__(self, config, shape):
        super().__init__()
        self.config = config
        self.shape = shape
        proj_dim = config["model"]["SimCLR"]["projection_dimension"]
        temperature = config["model"]["SimCLR"]["temperature"]

        backbone = models.resnet18(pretrained=False)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )
        self.criterion = NTXentLoss(temperature)

    def forward(self, x):
        feats = self.backbone(x)
        return self.projection_head(feats)

    def train_step(self, x):
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
