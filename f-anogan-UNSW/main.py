import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from fanogan_muti.train_wgangp import train_wgangp
from fanogan_muti.train_encoder_izif import train_encoder_izif
from model import Generator, Discriminator, Encoder
from tools import SimpleDataset, load_UNSW, NormalizeTransform
import numpy as np
import pandas as pd
from fanogan_muti.test_anomaly_detection import test_anomaly_detection
def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    (x_train, y_train), (x_test, y_test) = load_UNSW()
    
    mean = x_train.mean(axis=0)  # Mean of each feature
    std = x_train.std(axis=0)
    normalize = NormalizeTransform(mean, std)
    y_train = torch.zeros(len(x_train))
    train_mnist = SimpleDataset(x_train, y_train,transform=normalize)
    train_dataloader = DataLoader(train_mnist, batch_size=opt.batch_size,shuffle=True)
    gan_input_dim = x_train.shape[1]
    latent_dim = int(gan_input_dim * 0.5)

    generator = Generator(gan_input_dim, latent_dim)
    discriminator = Discriminator(gan_input_dim)
    train_wgangp(opt, generator, discriminator, train_dataloader, device, latent_dim)

    encoder = Encoder(gan_input_dim, latent_dim)
    train_encoder_izif(opt, generator, discriminator, encoder, train_dataloader, device)

    test_mnist = SimpleDataset(x_test, y_test,transform=normalize)
    test_dataloader = DataLoader(test_mnist, batch_size=1,shuffle=False)
    test_anomaly_detection(opt, generator, discriminator, encoder,
                        test_dataloader, device)
    


"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=80,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=8,
                        help="dimensionality of the latent space")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for "
                             "discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=3000,
                        help="interval betwen image samples")
    parser.add_argument("--split_rate", type=float, default=0.8,
                        help="rate of split for normal training data")
    parser.add_argument("--seed", type=int, default=42,
                        help="value of a random seed")
    opt = parser.parse_args()

    main(opt)
