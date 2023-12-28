import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, activation=nn.ReLU):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=5, kernel_size=3),
            activation(),
            nn.BatchNorm2d(5),

            nn.Conv2d(in_channels=5, out_channels=7, kernel_size=3, stride=3),
            activation(),
            nn.BatchNorm2d(7),

            nn.MaxPool2d(3),

            nn.Conv2d(in_channels=7, out_channels=5, kernel_size=3, stride=3),
            activation(),
            nn.BatchNorm2d(5),

            nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3, stride=2),
            activation(),
            nn.BatchNorm2d(3)
        )

        self.fc = nn.Sequential(
            nn.Linear(48, 38),
            nn.Linear(38, 28)
        )
        self.mu = nn.Linear(28, latent_dim)
        self.log_var = nn.Linear(28, latent_dim)
        self.flatten = nn.Flatten()

        # channels list consist of

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        normal_dist = torch.randn_like(mu)
        z = mu + normal_dist*std
        return z

    def forward(self, x: torch.Tensor):
        x = self.convs(x)
        print(x.shape, "shepu")
        x = self.flatten(x)
        x = self.fc(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self.reparametrize(mu, log_var)

        return z, mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 28),
            nn.Linear(28, 38),
            nn.Linear(38, 48)
        )

        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3, out_channels=5,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=5, out_channels=7,
                               kernel_size=3, stride=3, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=7, out_channels=5,
                               kernel_size=3, stride=3, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=5, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )
        # self.deconvs = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=3, out_channels=5,  # padding=1,
        #                        kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=5, out_channels=7,  # padding=1,
        #                        kernel_size=3, stride=3),
        #     nn.ReLU(),
        #     nn.Upsample(3),
        #     nn.ConvTranspose2d(in_channels=7, out_channels=5,  # padding=1,
        #                        kernel_size=3, stride=3),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=5, out_channels=out_channels,  # padding=1,
        #                        kernel_size=3, stride=1),

        # )

    def forward(self, z):
        z = self.fc(z)
        print(f"Z shape:{z.shape}")
        z = z.view(z.size(0), 3, 4, 4)
        x_hat = self.deconvs(z)
        return x_hat


if __name__ == "__main__":
    vae = Encoder(in_channels=3, latent_dim=20)
    dec = Decoder(latent_dim=20, out_channels=3)
    print(vae)
    random_tensor = torch.rand(32, 3, 250, 250)
    print(vae(random_tensor)[0].shape)
    print(vae(random_tensor)[1].shape)
    print(vae(random_tensor)[2].shape)
    z = vae(random_tensor)[0]
    ekesi = dec(z)
    print(ekesi.shape)

    # print(random_tensor.shape)
    # print(vae)
