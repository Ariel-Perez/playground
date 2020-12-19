import enum
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class Algorithm(enum.Enum):
    LINEAR = 0
    DNN = 1
    CNN = 2
    HIGHWAY_NETWORK = 3
    RESNET = 4


class Model(pl.LightningModule):

    def __init__(self):
        super().__init__()

    def loss(self, y_hat, y):
        return F.nll_loss(y_hat, y)

    def training_step(self, batch, _):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        pred = torch.argmax(y_hat, dim=1)
        accuracy = (pred == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', accuracy)
        return {'val_loss': loss, 'val_acc': accuracy}

    def test_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        pred = torch.argmax(y_hat, dim=1)
        accuracy = (pred == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', accuracy)
        return {'test_loss': loss, 'test_acc': accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        # return [optimizer], [scheduler]
        return optimizer


class Linear(Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(np.prod(input_dim), output_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


class DNN(Model):
    def __init__(self, input_dim, output_dim, num_layers):
        super().__init__()
        self.initial_layer = nn.Linear(np.prod(input_dim), 64)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(64, 64) for _ in range(num_layers - 1)
        ])
        self.final_layer = nn.Linear(64, np.prod(output_dim))

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.initial_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.final_layer(x)
        return F.log_softmax(x, dim=1)


class CNN(Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        depth, height, width = input_dim
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(depth, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ])
        self.fc_layers = nn.ModuleList([
            nn.Linear(512 * (height // 4) * (width // 4), 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        ])
        self.final_layer = nn.Linear(128, output_dim)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(x.shape[0], -1)
        for layer in self.fc_layers:
            x = layer(x)

        x = self.final_layer(x)
        return F.log_softmax(x, dim=1)


class HighwayNetwork(Model):

    def __init__(self, input_dim, output_dim, num_layers):
        super().__init__()
        depth, height, width = input_dim
        self.initial_layer = nn.Linear(height * width * depth, 50)
        self.hidden_layers = nn.ModuleList([
            self.HighwayLayer(50, 50, bias_init=-(num_layers // 10))
            for _ in range(num_layers - 1)
        ])
        self.final_layer = nn.Linear(50, output_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.initial_layer(x))
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.final_layer(x)
        return F.log_softmax(x, dim=1)

    class HighwayLayer(nn.Module):
        def __init__(self, input_units, output_units, bias_init=-1):
            super().__init__()
            self.h = nn.Linear(input_units, output_units)
            self.t = nn.Linear(input_units, output_units)
            nn.init.constant_(self.t.bias, bias_init)

        def forward(self, x):
            t_x = torch.sigmoid(self.t(x))
            return F.relu(self.h(x)) * t_x + x * (1 - t_x)


class ResidualNetwork(Model):
    def __init__(self, input_dim, output_dim, n=3):
        super().__init__()
        depth, _, _ = input_dim
        self.batch_norm = nn.BatchNorm2d(depth)
        self.initial_layer = nn.Conv2d(depth, 16, kernel_size=3, padding=1)
        self.first_residual_blocks = nn.ModuleList([
            self.ResidualBlock(16, 16)
            for i in range(2 * n)
        ])
        self.second_residual_blocks = nn.ModuleList(
            [self.ResidualBlock(16, 32)] + [
            self.ResidualBlock(32, 32)
            for i in range(2 * n - 1)
        ])
        self.third_residual_blocks = nn.ModuleList(
            [self.ResidualBlock(32, 64)] + [
                self.ResidualBlock(64, 64)
                for i in range(2 * n - 1)
        ])
        self.pool = nn.AvgPool2d(kernel_size=8)
        self.final_layer = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.initial_layer(x)
        for layer in self.first_residual_blocks:
            x = layer(x)

        for layer in self.second_residual_blocks:
            x = layer(x)

        for layer in self.third_residual_blocks:
            x = layer(x)

        x = self.pool(x)
        x = self.final_layer(x.squeeze())
        return F.log_softmax(x, dim=1)

    class ResidualBlock(nn.Module):
        def __init__(self, input_channels, output_channels):
            super().__init__()
            self.input_channels = input_channels
            self.output_channels = output_channels
            self.first_layer = nn.Conv2d(
                input_channels, output_channels, kernel_size=3, padding=1,
                stride=2 if output_channels > input_channels else 1)
            self.second_layer = nn.Conv2d(
                output_channels, output_channels, kernel_size=3, padding=1)
            self.identity_pool = nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(1, 2, 2))

        def forward(self, x):
            t = F.relu(self.first_layer(x))
            if self.output_channels > self.input_channels:
                x = self.identity_pool(x)
                x = F.pad(x, (0, 0, 0, 0, 0, self.input_channels))
            t = self.second_layer(t)
            return F.relu(t + x)
