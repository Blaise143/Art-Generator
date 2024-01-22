import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
import pytorch_lightning as pl
import yaml
from torch.optim import Adam
from pytorch_lightning.utilities.model_summary import ModelSummary


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        channels_order = [config['model_config']['input_channels']] + \
            [layer['out_channels']
                for layer in config['model_config']['encoder_layers']]

        latent_dim = config['model_config']['latent_dim']
        input_size = config['model_config']['input_size']

        # Encoder
        encode_layers = []
        for i in range(len(channels_order) - 1):
            encode_layers.extend(
                [
                    nn.Conv2d(
                        channels_order[i], channels_order[i + 1],
                        kernel_size=config['model_config']['encoder_layers'][i]['kernel_size'],
                        stride=config['model_config']['encoder_layers'][i]['stride'],
                        padding=config['model_config']['encoder_layers'][i].get(
                            'padding', 0)
                    ),
                    nn.BatchNorm2d(channels_order[i+1]),
                    nn.ReLU(),
                    # nn.MaxPool2d(2),
                    nn.Dropout(config['training_config']['dropout'])
                ]
            )
        self.encoder = nn.Sequential(*encode_layers)

        # Calculate the output size of the last conv layer to set input size for the first linear layer
        mlp_input_dim, self.decoder_input_shape = self._calc_flatten_dim(
            input_size, channels_order[-1])

        self.mu = nn.Linear(mlp_input_dim, latent_dim)
        self.log_var = nn.Linear(mlp_input_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, mlp_input_dim)

        # Decoder
        decode_layers = []
        channels_order.reverse()
        for i in range(len(channels_order) - 2):
            decode_layers.extend(
                [
                    nn.ConvTranspose2d(
                        channels_order[i], channels_order[i + 1],
                        kernel_size=config['model_config']['encoder_layers'][i]['kernel_size'],
                        stride=config['model_config']['encoder_layers'][i]['stride'],
                        padding=config['model_config']['encoder_layers'][i].get(
                            'padding', 1),
                        output_padding=config['model_config']['encoder_layers'][i].get(
                            'output_padding', 0)
                    ),
                    # nn.BatchNorm2d(channels_order[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(config['training_config']['dropout'])
                ]
            )
        i = -2
        decode_layers.extend(
            [
                nn.ConvTranspose2d(
                    channels_order[i], channels_order[i+1],
                    kernel_size=config['model_config']['encoder_layers'][i]['kernel_size'],
                    stride=config['model_config']['encoder_layers'][i]['stride'],
                    padding=config['model_config']['encoder_layers'][i].get(
                        'padding', 1),
                    output_padding=config['model_config']['encoder_layers'][i].get(
                        'output_padding', 0)),
                nn.Sigmoid()
            ]
        )
        self.decoder = nn.Sequential(*decode_layers)

    def _calc_flatten_dim(self, input_size, out_channels):
        # Dummy variable to calculate output size
        x = torch.rand(1, self.config['model_config']
                       ['input_channels'], *input_size)
        with torch.no_grad():
            x = self.encoder(x)
        return int(torch.flatten(x, 1).size(1)), x.size()[1:]

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.decoder_input(z))
        z = z.view(z.size(
            0), self.decoder_input_shape[0], self.decoder_input_shape[1], self.decoder_input_shape[2])
        z = self.decoder(z)
        return z

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def _step(self, batch, step_type="train"):
        x, _ = batch
        recon_x, mu, log_var = self.forward(x)

        # Calculate loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        if step_type == "train":
            beta = 0.8
        else:
            beta = 1.0

        loss = recon_loss + beta * kld_loss

        return loss, recon_loss, kld_loss

    def training_step(self, batch, batch_idx):
        loss, recon_loss, kld_loss = self._step(batch, step_type="train")

        self.log('train_loss', loss)
        self.log('train_recon_loss', recon_loss)
        self.log('train_kld_loss', kld_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, kld_loss = self._step(batch, step_type="val")

        self.log('val_loss', loss)
        self.log('val_recon_loss', recon_loss)
        self.log('val_kld_loss', kld_loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=self.config['training_config']['learning_rate'])
        return optimizer


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    print(config)
    x = torch.randn((1, 3, 250, 250))
    print(x.shape)
    vae = VariationalAutoEncoder(config)
    # print("SHAPE")
    print(vae(x)[0].shape)
    # print(vae)
    summary = ModelSummary(vae)
    print(summary)
