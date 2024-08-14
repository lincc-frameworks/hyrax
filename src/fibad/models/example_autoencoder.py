# ruff: noqa: D101, D102

# This example model is taken from the autoenocoder tutorial here
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa N812
import torch.optim as optim
import torch.utils.data.dataloader

# extra long import here to address a circular import issue
from fibad.models.model_registry import fibad_model


@fibad_model
class ExampleAutoencoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config

        self.c_hid = self.config.get("base_channel_size", 32)
        self.num_input_channels = self.config.get("num_input_channels", 3)
        self.latent_dim = self.config.get("latent_dim", 64)
        self.act_fn = nn.GELU

        self._init_encoder()
        self._init_decoder()

    def _init_encoder(self):
        c_hid = self.c_hid
        num_input_channels = self.num_input_channels
        latent_dim = self.latent_dim
        act_fn = self.act_fn
        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, latent_dim),
        )

    def _eval_encoder(self, x):
        return self.encoder(x)

    def _init_decoder(self):
        num_input_channels = self.num_input_channels
        latent_dim = self.latent_dim
        act_fn = self.act_fn
        c_hid = self.c_hid

        self.dec_linear = nn.Sequential(nn.Linear(latent_dim, 2 * 16 * c_hid), act_fn())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 16x16 => 32x32
            nn.Tanh(),  # The input images is scaled between -1 and 1, so the output has to be bounded as well
        )

    def _eval_decoder(self, x):
        x = self.dec_linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.decoder(x)
        return x

    def forward(self, x):
        z = self._eval_encoder(x)
        x_hat = self._eval_decoder(z)
        return x_hat

    def train(self, trainloader, device=None):
        self.optimizer = self._optimizer()

        torch.set_grad_enabled(True)

        for epoch in range(self.config.get("epochs", 2)):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                x, _ = data  # This discards labels from CIFAR10 dataset. May need to pull for HSC data

                x = x.to(device)
                x_hat = self.forward(x)
                loss = F.mse_loss(x, x_hat, reduction="none")
                loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
                    running_loss = 0.0

    def _optimizer(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def save(self):
        torch.save(self.state_dict(), self.config.get("weights_filepath", "example_autoencoder.pth"))
