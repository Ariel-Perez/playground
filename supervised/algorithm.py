import enum
import torch
import torch.nn as nn
import torch.nn.functional as F


class Algorithm(enum.Enum):
    LINEAR = 0
    DNN = 1
    CNN = 2
    SLIM = 3


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.linear(x)


class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.hidden_layers = nn.ModuleList([
            nn.Linear(input_dim, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
        ])
        self.final_layer = nn.Linear(256, output_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.final_layer(x)


class CNN(nn.Module):
    def __init__(self, size, output_dim):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ])
        self.fc_layers = nn.ModuleList([
            nn.Linear(512 * 7 * 7, 128),
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

        return self.final_layer(x)


class Slim(nn.Module):
    def __init__(self, size, output_dim):
        super().__init__()
        height, width = size
        self.horizontal = nn.Conv2d(1, 256, kernel_size=(3, width))
        self.horizontal_2 = nn.Conv2d(256, 512, kernel_size=(1, width - 2))
        self.vertical = nn.Conv2d(1, 256, kernel_size=(height, 3))
        self.vertical_2 = nn.Conv2d(256, 512, kernel_size=(height - 2, 1))
        self.fc_layers = nn.ModuleList([
            nn.Linear(1024, 256),
            nn.Linear(256, 128),
        ])
        self.final_layer = nn.Linear(128, output_dim)

    def forward(self, x):
        horizontal = F.relu(self.horizontal(x))
        vertical = F.relu(self.vertical(x))
        x = torch.cat([
            F.relu(self.vertical_2(horizontal)).squeeze(),
            F.relu(self.horizontal_2(vertical)).squeeze()], dim=1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))

        return self.final_layer(x)