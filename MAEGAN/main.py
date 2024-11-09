import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model.gan import Generator, Discriminator, Encoder
from tools import SimpleDataset, load_UGR16, NormalizeTransform, report_result
import model.MAE as mae
import numpy as np
import pandas as pd
from model.MAEGAN import MAEGAN


from sklearn.preprocessing import StandardScaler, MinMaxScaler
def load_cic2017_g():
    seq_len = 5
    embs_path = "/root/GCN/DyGCN/data/data/cic2017/model-DGC5-2.pt"
    labels_path = "/root/GCN/DyGCN/data/data/cic2017/labels.npy"
    train_len=[0, 527]
    data_embs = torch.load(embs_path).detach().cpu().numpy()
    x_train = data_embs[train_len[0]+seq_len:train_len[1]]
    x_test=np.concatenate([data_embs[:train_len[0]],data_embs[train_len[1]:]])
    y_train = np.zeros(len(x_train))
    labels = np.load(labels_path, allow_pickle=True)
    labels = labels.astype(int)
    labels=labels[seq_len:]
    y_test=np.concatenate((labels[:train_len[0]], labels[train_len[1]:]))
    minmax_scaler = MinMaxScaler()
    x_train = minmax_scaler.fit_transform(x_train)  # 仅在训练数据上拟合
    x_test = minmax_scaler.transform(x_test)  # 使用相同的缩放器进行转换
    return (x_train, y_train), (x_test, y_test)

def load_ugr16_g():
    seq_len = 5
    embs_path = "/root/GCN/DyGCN/data/data/ugr16/model-DGC5-2.pt"
    labels_path = "/root/GCN/DyGCN/data/data/ugr16/labels.npy"
    train_len=[0, 500]
    data_embs = torch.load(embs_path).detach().cpu().numpy()
    x_train = data_embs[train_len[0]+seq_len:train_len[1]]
    x_test=np.concatenate([data_embs[:train_len[0]],data_embs[train_len[1]:]])
    y_train = np.zeros(len(x_train))
    labels = np.load(labels_path, allow_pickle=True)
    labels = labels.astype(int)
    labels=labels[seq_len:]
    y_test=np.concatenate((labels[:train_len[0]], labels[train_len[1]:]))
    minmax_scaler = MinMaxScaler()
    x_train = minmax_scaler.fit_transform(x_train)  # 仅在训练数据上拟合
    x_test = minmax_scaler.fit_transform(x_test)  # 使用相同的缩放器进行转换
    return (x_train, y_train), (x_test, y_test)

def load_UGR16():
    raw_x_train = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)
    x_train = raw_x_train
    x_train = torch.from_numpy(x_train.values).float()
    y_train = torch.zeros(len(x_train))


    raw_x_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
    x_test = raw_x_test
    x_test = torch.from_numpy(x_test.values).float()
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    y_test = torch.from_numpy(y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values)

    # 接下来进行归一化
    (x_train, y_train), (x_test, y_test) = (x_train.numpy(), y_train.numpy()), (x_test.numpy(), y_test.numpy())
    minmax_scaler = MinMaxScaler()
    x_train = minmax_scaler.fit_transform(x_train)  # 仅在训练数据上拟合
    x_test = minmax_scaler.transform(x_test)  # 使用相同的缩放器进行转换
    return (x_train, y_train), (x_test, y_test)

def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # (x_train, y_train), (x_test, y_test) = load_UNSW()
    (x_train, y_train), (x_test, y_test) = load_cic2017_g()
    print(x_train.shape)
    filepath = "ugr16"
    maegan = MAEGAN(opt, input_dim = x_train.shape[1], filepath=filepath, batch_size=4)
    print("Running KitNET:")
    maegan.train(x_train)
    score = maegan.test(x_test, y_test)
    report_result(name=filepath, anomaly_score=score, labels=y_test)
    



"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=40,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.99,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.99,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for "
                             "discriminator per iter")
    parser.add_argument("--seed", type=int, default=42,
                        help="value of a random seed")
    opt = parser.parse_args()

    main(opt)
