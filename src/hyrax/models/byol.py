# ruff: noqa: D101, D102


import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa N812
import torchvision.models as models
import torchvision.transforms.v2 as T  # noqa N812
from byol_pytorch import BYOL

from hyrax.models.model_registry import hyrax_model


class ArcsinhActivation(nn.Module):
    """Helper module for HyraxAutoencoderV2 to use the arcsinh function"""

    def forward(self, x):
        return torch.arcsinh(x)


@hyrax_model
class HyraxBYOL(nn.Module):
    def __init__(self, config, shape):
        super().__init__()
        self.config = config
        self.shape = shape
        self.siamese = config["model"]["BYOL"]["siamese"]

        # backbone = models.resnet18(pretrained=False)
        backbone = models.resnet34(pretrained=True)

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

        backbone.fc = self.final_activation
        self.backbone = backbone

        # The wrapper written by the original authors
        # The predictor has the same final output size and hidden size as the projector
        self.learner = BYOL(
            self.backbone,
            image_size=self.shape[-1],
            hidden_layer="avgpool",
            projection_size=config["model"]["BYOL"]["projection_size"],
            projection_hidden_size=config["model"]["BYOL"]["projection_hidden_size"],
            moving_average_decay=config["model"]["BYOL"]["moving_average_decay"],
            use_momentum=self.siamese,
        )

    def forward(self, x):
        # Only perform forward pass using just the model (encoder), not the wrapper
        return self.backbone(x)

    def train_batch(self, x):
        # The wrapper automatically performs loss computation
        loss = self.learner(x)

        # Back propagation routine
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target model if not using SimSiam variant
        if not self.siamese:
            self.learner.update_moving_average()

        return {"loss": loss.item()}
    
    def validate_batch(self, x):
        # The wrapper automatically performs loss computation
        loss = self.learner(x)
        return {"loss": loss.item()}
    
    def test_batch(self, x):
        return self.validate_batch(x)

    def infer_batch(self, x):
        return self.validate_batch(x)

    def _optimizer(self):
        return torch.optim.Adam(self.learner.parameters(), lr=3e-4)
