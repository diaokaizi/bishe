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
    # feature_map =  [[113], [34, 86], [64], [2, 5], [27, 79], [33, 85], [15, 67], [101], [42, 94], [0, 3], [60], [47, 99], [37, 89], [51], [14, 66], [44, 96], [63], [11, 12], [46, 98], [13, 65], [77], [53, 105, 50, 102], [78], [90], [48, 100], [45, 97], [26], [122], [49, 36, 88, 29, 81, 114, 58, 6, 112], [7], [43, 95], [59], [24, 76], [39, 91], [41, 93], [103], [28, 80], [123], [118, 17, 69], [18, 70, 111], [131], [128], [124, 129], [130], [31, 83], [132], [126, 127, 133], [22, 74], [56, 108], [57, 109, 121, 55, 107, 116, 1, 4], [110, 119, 125, 117, 120], [52, 104, 32, 84, 16, 68], [54, 106], [62], [75], [82], [73], [30, 23, 87, 25, 35, 8, 9, 21, 10, 38], [61, 71, 72, 19, 20, 115], [40, 92]]
    feature_map = [[32, 68, 16, 113, 84], [32, 34, 68, 16, 86], [64, 38, 10, 77, 25], [2, 5, 70, 111, 18], [6, 79, 114, 58, 27], [33, 70, 111, 85, 123], [1, 67, 4, 107, 15], [101, 81, 49, 88, 29], [4, 42, 108, 56, 94], [0, 3, 70, 111, 18], [71, 72, 73, 60, 61], [99, 47, 117, 120, 125], [37, 70, 111, 89, 123], [133, 103, 51, 126, 127], [66, 69, 14, 17, 118], [96, 133, 44, 126, 127], [133, 41, 63, 93, 127], [8, 9, 10, 11, 12], [98, 132, 133, 46, 126], [65, 133, 13, 126, 127], [38, 9, 10, 77, 25], [102, 105, 50, 53, 29], [122, 103, 78, 22, 26], [38, 8, 9, 10, 90], [128, 100, 133, 48, 127], [97, 1, 4, 45, 116], [122, 103, 74, 22, 26], [103, 74, 110, 22, 122], [36, 6, 112, 49, 81, 114, 88, 58, 29], [32, 7, 84, 122, 59], [133, 43, 95, 126, 127], [32, 68, 16, 84, 59], [1, 76, 109, 24, 57], [133, 39, 91, 126, 127], [133, 41, 93, 126, 127], [133, 103, 120, 126, 127], [133, 109, 80, 28, 127], [70, 111, 18, 116, 123], [130, 69, 17, 118, 119], [70, 107, 111, 18, 116], [131, 116, 117, 120, 125], [128, 133, 120, 126, 127], [129, 1, 4, 121, 124], [130, 117, 119, 120, 125], [109, 83, 117, 120, 31], [132, 133, 117, 120, 126], [133, 117, 120, 126, 127], [74, 110, 117, 22, 120], [1, 4, 108, 116, 56], [121, 1, 4, 107, 109, 116, 55, 57], [110, 117, 119, 120, 125], [32, 68, 104, 16, 52, 84], [106, 83, 54, 120, 31], [38, 8, 9, 10, 62], [38, 10, 75, 21, 23], [38, 9, 10, 82, 30], [71, 72, 73, 61, 62], [35, 38, 8, 9, 10, 23, 21, 87, 25, 30], [71, 72, 19, 20, 115, 61], [103, 40, 22, 122, 92]]
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
    mean = gsa.mean(axis=0)  # Mean of each feature
    std = gsa.std(axis=0)
    normalize = NormalizeTransform(mean, std)
    y_train = torch.zeros(len(gsa))
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
