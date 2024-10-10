import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

"""
The code is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.scale = 1.0 / (in_dim ** 0.5)

    def forward(self, x):
        # x: (batch_size, in_dim)
        Q = self.query(x)  # (batch_size, in_dim)
        K = self.key(x)    # (batch_size, in_dim)
        V = self.value(x)  # (batch_size, in_dim)

        # 将特征维度视为序列长度
        Q = Q.unsqueeze(1)  # (batch_size, 1, in_dim)
        K = K.unsqueeze(2)  # (batch_size, in_dim, 1)
        V = V.unsqueeze(1)  # (batch_size, 1, in_dim)

        # 计算注意力得分
        attn_scores = torch.matmul(Q, K) * self.scale  # (batch_size, 1, 1)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, 1, 1)

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, 1, in_dim)
        attn_output = attn_output.squeeze(1)         # (batch_size, in_dim)

        # 残差连接
        output = x + attn_output  # (batch_size, in_dim)
        return output

class Generator(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 55, normalize=False),
            SelfAttention(55),
            *block(55, 60),
            SelfAttention(60),
            *block(60, 65),
            SelfAttention(65),
            *block(65, 70),
            SelfAttention(70),
            *block(70, 75),
            SelfAttention(75),
            nn.Linear(75, self.input_dim),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim

        self.features = nn.Sequential(
            nn.Linear(self.input_dim, 16),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(16),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(8)
        )

        self.last_layer = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid()
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
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 75),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(75),
            nn.Linear(75, 70),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(70),
            nn.Linear(70, 65),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(65),
            nn.Linear(65, 60),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(60),
            nn.Linear(60, 55),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(55),
            nn.Linear(55, self.latent_dim),
            nn.Tanh()
        )

    def forward(self, img):
        latent = self.model(img)
        return latent
