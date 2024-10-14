import numpy as np
import torch.nn as nn


"""
The code is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


class Generator(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 60, normalize=False),
            *block(60, 65),
            *block(65, 70),
            *block(70, 75),
            *block(75, 80),
            *block(80, 85),
            *block(85, 90),
            *block(90, 95),
            nn.Linear(95, self.input_dim),
            nn.Tanh()
            )

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        self.features = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.2, inplace=True)
            )

        self.last_layer = nn.Sequential(
            nn.Linear(8, 1)
            )

    def forward(self, img):
        features = self.forward_features(img)
        validity = self.last_layer(features)
        return validity

    def forward_features(self, img):
        features = self.features(img)
        return features


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 95),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(95, 90),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(90, 85),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(85, 80),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(80, 75),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(75, 70),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(70, 65),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(65, 60),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(60, self.latent_dim),
            nn.Tanh()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity
