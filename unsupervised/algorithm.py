import enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Algorithm(enum.Enum):
    AUTOENCODER = 0
    VAE = 1


class VariationalAutoEncoder(nn.Module):
    def __init__(self, dimensions, embedding_dim=64,
                 hidden_layers=[32, 64, 128, 256]):
        super().__init__()
        self.dimensions = dimensions
        self.embedding_dim = embedding_dim
        self.encoder = Encoder(
            dimensions, embedding_dim * 2, hidden_layers)
        self.decoder = Decoder(
            dimensions, embedding_dim,
            list(reversed(hidden_layers)))
        self._initialize_weights()

    def loss(self, o, x, mu, log_var, kl_weight):
        reconstruction_loss = F.mse_loss(o, x)
        # kl_loss = torch.mean(
        #     -0.5 * torch.sum(
        #         1 + log_var - mu ** 2 - torch.exp(log_var), dim = 1),
        #     dim = 0,
        # )
        return reconstruction_loss  # + kl_weight * kl_loss

    def forward(self, x):
        x = self.encoder(x)
        mean, log_var = x.split(self.embedding_dim, dim=1)
        std = torch.exp(0.5 * log_var)
        x = mean + std * torch.randn_like(std)
        x = self.decoder(x)
        return x, mean, log_var

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class AutoEncoder(nn.Module):
    def __init__(self, dimensions, embedding_dim=64,
                 hidden_layers=[32, 64, 128, 256]):
        super().__init__()
        self.dimensions = dimensions
        self.embedding_dim = embedding_dim
        self.encoder = Encoder(
            dimensions, embedding_dim, hidden_layers)
        self.decoder = Decoder(
            dimensions, embedding_dim,
            list(reversed(hidden_layers)))
        self._initialize_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Encoder(nn.Module):
    def __init__(self, dimensions, embedding_dim, hidden_layers):
        super().__init__()
        depth, height, width = dimensions
        self.dimensions = dimensions
        self.embedding_dim = embedding_dim
        channels = [depth] + hidden_layers
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ) for i in range(len(hidden_layers))
        ])
        for i in range(len(hidden_layers)):
            height = (height + 1) // 2
            width = (width + 1) // 2
        final_dimensions = (hidden_layers[-1], height, width)
        self.flatten = nn.Flatten()
        self.output_layer = nn.Linear(np.prod(final_dimensions), embedding_dim)

    def forward(self, x):
        x = self.blocks(x)
        x = self.flatten(x)
        x = self.output_layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dimensions, embedding_dim, hidden_layers):
        super().__init__()
        depth, height, width = dimensions
        self.dimensions = dimensions
        self.embedding_dim = embedding_dim
        for i in range(len(hidden_layers)):
            height = (height + 1) // 2
            width = (width + 1) // 2
        self.initial_dimensions = (hidden_layers[0], height, width)
        self.reassemble = nn.Linear(
            embedding_dim, np.prod(self.initial_dimensions))
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_layers[i - 1], hidden_layers[i], kernel_size=3,
                    padding=1, stride=2, output_padding=1),
                nn.BatchNorm2d(hidden_layers[i]),
                nn.ReLU(inplace=True),
            ) for i in range(1, len(hidden_layers))
        ])
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_layers[-1], hidden_layers[-1], kernel_size=3,
                               padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(hidden_layers[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_layers[-1], depth, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.reassemble(x)
        x = x.view(batch_size, *self.initial_dimensions)
        x = self.blocks(x)
        x = self.output_layer(x)
        if x.shape[-1] > self.dimensions[-1]:
            x = x.narrow(-1, 0, self.dimensions[-1])
        if x.shape[-2] > self.dimensions[-2]:
            x = x.narrow(-2, 0, self.dimensions[-2])
        return x
