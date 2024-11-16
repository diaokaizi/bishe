import numpy as np
import torch.nn as nn


"""
The code is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""

class Autoencoder(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=10):
        super(Autoencoder, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 130),
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
            nn.Linear(70, hidden_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            *block(hidden_dim, 70, normalize=False),
            *block(70, 80),
            *block(80, 90),
            *block(90, 100),
            *block(100, 110),
            *block(110, 120),
            *block(120, 130),
            nn.Linear(130, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
