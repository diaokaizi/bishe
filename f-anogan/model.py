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
            *block(self.latent_dim, 70, normalize=False),
            *block(70, 80),
            *block(80, 90),
            *block(90, 100),
            *block(100, 110),
            *block(110, 120),
            *block(120, 130),
            nn.Linear(130, self.input_dim),
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
            nn.Linear(self.input_dim, 16),
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
            nn.Linear(self.input_dim, 75),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(130, 120),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(120, 110),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(110, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 90),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(90, 80),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(80, 70),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(70, self.latent_dim),
            nn.Tanh()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity
