import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from fanogan_muti.train_wgangp import train_wgangp
from fanogan_muti.train_encoder_izif import train_encoder_izif
from model import Generator, Discriminator, Encoder
from tools import SimpleDataset, load_UGR16, NormalizeTransform
import KitNET as kit
import numpy as np
import pandas as pd
from fanogan_muti.test_anomaly_detection import test_anomaly_detection
def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_train = pd.read_csv("/root/UGR16_FeatureData/csv/UGR16v1.Xtrain.csv").drop(columns=['Row'], axis=1).values #an m-by-n dataset with m observations
    # train_mnist = SimpleDataset(x_train, y_train)
    maxAE = 10 #maximum size for any autoencoder in the ensemble layer
    # Build KitNET
    feature_map =  [[113], [64], [2, 5], [34, 86], [51], [40, 92], [42, 94], [35, 87], [27, 79], [30, 82], [44, 96], [75], [23], [16, 68], [13, 65], [60], [52], [115], [25, 77], [38], [14, 66], [33, 85], [21, 73], [0, 3], [12], [46, 98], [48, 100], [90], [49, 101], [47, 99], [9, 61, 8, 11, 10, 62], [104], [63], [71, 72, 19, 20], [53, 105, 50, 102], [45, 97], [39, 91], [43, 95], [7, 59], [106], [32, 36, 88], [78], [29], [37, 89], [67], [15], [26], [54], [6, 112, 58, 114], [122], [41, 93], [81, 84], [123], [118, 17, 69], [103], [28, 80, 24, 76], [128], [127], [18, 70], [130], [129], [74], [83], [133, 126, 132, 125, 117, 120, 110, 119, 55, 107], [22, 31], [131], [111, 124, 56, 108, 57, 109, 121, 116, 1, 4]]
    # feature_map = [[2, 5], [34, 86], [40, 92], [42, 94], [35, 87], [27, 79], [30, 82], [44, 96], [16, 68], [13, 65], [25, 77], [38], [14, 66], [33, 85], [21, 73], [0, 3], [46, 98], [48, 100], [49, 101], [47, 99], [9, 61, 8, 11, 10, 62], [104], [71, 72, 19, 20], [53, 105, 50, 102], [45, 97], [39, 91], [43, 95], [7, 59], [32, 36, 88], [37, 89], [6, 112, 58, 114], [41, 93], [81, 84], [118, 17, 69], [28, 80, 24, 76], [18, 70], [74], [133, 126, 132, 125, 117, 120, 110, 119, 55, 107], [22, 31], [111, 124, 56, 108, 57, 109, 121, 116, 1, 4]]
    # feature_map = [16, 68], [35, 87], [37, 89], [38, 90], [40, 92], [45, 97], [55, 107], [57, 109], [22, 74], [33, 85], [44, 96], [52, 104], [116, 117, 126]
    print(len(feature_map))
    K = kit.KitNET(x_train.shape[1],maxAE,0,0, feature_map=feature_map)
    print("Running KitNET:")
    # Here we process (train/execute) each individual observation.
    # In this way, X is essentially a stream, and each observation is discarded after performing process() method.


    gsa = np.zeros([x_train.shape[0], len(feature_map)]) # a place to save the scores
    for epo in range(2):
        for i in range(x_train.shape[0]):
            if i % 1000 == 0:
                print(epo, i)
            gsa[i] = K.train(x_train[i,]) #will train during the grace periods, then execute on all the rest.
        pd.DataFrame(gsa).to_csv("gsa.csv", index=False)
    
    (a, b), (c, _) = load_UGR16()
    
    
    print("Running fanogan:")
    gsa = torch.from_numpy(gsa).float()
    gsa = torch.cat([gsa, a], dim=1)
    print(gsa.shape)
    y_train = torch.zeros(len(gsa))
    mean = gsa.mean(axis=0)  # Mean of each feature
    std = gsa.std(axis=0)
    normalize = NormalizeTransform(mean, std)
    train_mnist = SimpleDataset(gsa, y_train,transform=normalize)
    train_dataloader = DataLoader(train_mnist, batch_size=opt.batch_size,shuffle=True)
    gan_input_dim = gsa.shape[1]
    # latent_dim = int(len(feature_map) * 0.5)
    latent_dim = 50

    generator = Generator(gan_input_dim, latent_dim)
    discriminator = Discriminator(gan_input_dim)
    train_wgangp(opt, generator, discriminator, train_dataloader, device, latent_dim)

    encoder = Encoder(gan_input_dim, latent_dim)
    train_encoder_izif(opt, generator, discriminator, encoder, train_dataloader, device)


    x_test = pd.read_csv("/root/UGR16_FeatureData/csv/UGR16v1.Xtest.csv").drop(columns=['Row'], axis=1).values
    gsa = np.zeros([x_test.shape[0], len(feature_map)])
    for i in range(x_test.shape[0]):
        if i % 1000 == 0:
            print(i)
        gsa[i] = K.execute(x_test[i,]) #will train during the grace periods, then execute on all the rest.
    gsa = torch.from_numpy(gsa).float()
    y_test = pd.read_csv("/root/UGR16_FeatureData/csv/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    y_test = torch.from_numpy(y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values)
    gsa = torch.cat([gsa, c], dim=1)
    mean = gsa.mean(axis=0)  # Mean of each feature
    std = gsa.std(axis=0)
    normalize = NormalizeTransform(mean, std)
    test_mnist = SimpleDataset(gsa, y_test,transform=normalize)
    test_dataloader = DataLoader(test_mnist, batch_size=1,shuffle=False)
    test_anomaly_detection(opt, generator, discriminator, encoder,
                        test_dataloader, device)
    
    # (x_train, y_train), _ = load_gas()
    # print(x_train.shape)
    # mean = x_train.mean(axis=0)  # Mean of each feature
    # std = x_train.std(axis=0)
    # normalize = NormalizeTransform(mean, std)
    # train_mnist = SimpleDataset(x_train, y_train,transform=normalize)
    # # train_mnist = SimpleDataset(x_train, y_train)
    # train_dataloader = DataLoader(train_mnist, batch_size=opt.batch_size,shuffle=True)
    # generator = Generator(x_train.shape[1])
    # discriminator = Discriminator(x_train.shape[1])
    # train_wgangp(opt, generator, discriminator, train_dataloader, device)


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
