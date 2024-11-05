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

from sklearn.preprocessing import StandardScaler, MinMaxScaler
# def load_UNSW():
#     # 先进行标准化
#     standard_scaler = StandardScaler()
    
#     # 加载训练数据
#     train = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_train_1s.csv")
    
#     # 计算每个样本的 anomaly_ratio 并筛选出 anomaly_ratio < 0.15 的样本
#     train['total_records'] = train['binary_label_normal'] + train['binary_label_attack']
#     train['anomaly_ratio'] = train['binary_label_attack'] / train['total_records']
#     train = train[train['anomaly_ratio'] < 0.15]  # 只保留 anomaly_ratio < 0.15 的样本

#     # 删除不需要的列
#     raw_x_train = train.drop(columns=['timestamp', 'label_background', 'label_exploits', 'label_fuzzers',
#                                        'label_reconnaissance', 'label_dos', 'label_analysis', 
#                                        'label_backdoor', 'label_shellcode', 'label_worms', 
#                                        'label_other', 'binary_label_normal', 'binary_label_attack', 
#                                        'total_records', 'anomaly_ratio'], axis=1)

#     # 标准化
#     x_train_standardized = standard_scaler.fit_transform(raw_x_train.values)  # 仅在训练数据上拟合

#     # 加载测试数据
#     raw_x_test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv").drop(columns=['timestamp', 
#                                        'label_background', 'label_exploits', 'label_fuzzers', 
#                                        'label_reconnaissance', 'label_dos', 'label_analysis', 
#                                        'label_backdoor', 'label_shellcode', 'label_worms', 
#                                        'label_other', 'binary_label_normal', 'binary_label_attack'], axis=1)

#     # 对测试数据进行标准化
#     x_test_standardized = standard_scaler.transform(raw_x_test.values)  # 使用相同的缩放器进行转换

#     # 接下来进行归一化
#     minmax_scaler = MinMaxScaler()
    
#     # 对标准化后的训练数据进行归一化
#     x_train_normalized = minmax_scaler.fit_transform(x_train_standardized)  # 仅在训练数据上拟合
    
#     # 对标准化后的测试数据进行归一化
#     x_test_normalized = minmax_scaler.transform(x_test_standardized)  # 使用相同的缩放器进行转换
    
#     # 加载并处理测试标签
#     y_test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv")
#     y_test['total_records'] = y_test['binary_label_normal'] + y_test['binary_label_attack']
#     y_test['anomaly_ratio'] = y_test['binary_label_attack'] / y_test['total_records']
    
#     # 根据 anomaly_ratio 生成测试标签
#     y_test = (y_test['anomaly_ratio'] > 0.15).astype(int).to_numpy()
    
#     # 假设训练数据全部为正常数据
#     y_train = np.zeros(len(x_train_normalized))

#     # 输出训练和测试集的形状
#     print(f"Training set shape: {x_train_normalized.shape}, Labels: {np.unique(y_train)}")
#     print(f"Test set shape: {x_test_normalized.shape}, Labels: {np.unique(y_test)}")
    
#     return (x_train_normalized, y_train), (x_test_normalized, y_test)




def load_group_np():
    seq_len=5
    embs_path = "/root/bishe/ae/graph_embs_0.809.pt"
    # embs_path='data/graph_embs.pt'
    labels_path = "/root/bishe/ae/labels.npy"

    train_len=[0, 500]

    data_embs = torch.load(embs_path).detach().cpu().numpy()
    print(len(data_embs))
    print(data_embs.shape)
    print(data_embs[train_len[0]+seq_len:train_len[1]].shape)
    print(np.concatenate([data_embs[:train_len[0]],data_embs[train_len[1]:]]).shape)
    labels = np.load(labels_path, allow_pickle=True)
    labels=labels[seq_len:]
    labels=np.concatenate((labels[:train_len[0]], labels[train_len[1]:]))
    print(labels)
    x_train = data_embs[train_len[0]+seq_len:train_len[1]]
    test_embs=np.concatenate([data_embs[:train_len[0]],data_embs[train_len[1]:]])
    x_test = test_embs
    y_test = labels
    y_train = np.zeros(len(x_train))
    return (x_train, y_train), (x_test, y_test)

def load_group_torch():
    seq_len=5
    embs_path = "/root/bishe/ae/graph_embs_0.809.pt"
    # embs_path='data/graph_embs.pt'
    labels_path = "/root/bishe/ae/labels.npy"

    train_len=[0, 500]

    data_embs = torch.load(embs_path).detach().cpu().numpy()
    print(len(data_embs))
    print(data_embs.shape)
    print(data_embs[train_len[0]+seq_len:train_len[1]].shape)
    print(np.concatenate([data_embs[:train_len[0]],data_embs[train_len[1]:]]).shape)
    labels = np.load(labels_path, allow_pickle=True)
    labels=labels[seq_len:]
    labels=np.concatenate((labels[:train_len[0]], labels[train_len[1]:]))
    print(labels)
    x_train = torch.from_numpy(data_embs[train_len[0]+seq_len:train_len[1]])
    test_embs=np.concatenate([data_embs[:train_len[0]],data_embs[train_len[1]:]])
    x_test = torch.from_numpy(test_embs)
    y_test = torch.from_numpy(labels)
    y_train = torch.zeros(len(x_train))
    return (x_train, y_train), (x_test, y_test)
    

def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (x_train, y_train), (x_test, y_test) = load_group_np()
    # train_mnist = SimpleDataset(x_train, y_train)
    maxAE = 10 #maximum size for any autoencoder in the ensemble layer
    # Build KitNET
    feature_map = [[7, 10, 11, 15, 21, 23, 26, 27, 28, 31], [0, 1, 2, 12, 13, 14, 16, 17, 18, 29], [5, 6, 7, 10, 11, 15, 21, 23, 26, 27], [0, 1, 2, 12, 13, 14, 16, 17, 18, 20], [0, 1, 2, 12, 13, 14, 16, 17, 18, 30], [1, 2, 3, 4, 6, 17, 18, 19, 20, 22], [7, 8, 9, 11, 15, 23, 24, 25, 27, 31], [7, 8, 10, 11, 15, 23, 24, 26, 27, 31]]
    # feature_map = [[10, 11, 26, 27, 28], [0, 12, 13, 14, 29], [5, 10, 11, 21, 26], [2, 12, 13, 17, 18], [0, 1, 14, 16, 30], [1, 2, 3, 4, 6, 17, 18, 19, 20, 22], [8, 9, 23, 24, 25], [7, 8, 10, 11, 15, 23, 24, 26, 27, 31]]
    # feature_map = [[28], [29], [5, 21], [12, 13], [0, 16, 14, 30], [19, 20, 3, 4, 6, 1, 17, 22, 2, 18], [9, 25], [15, 31, 8, 24, 10, 26, 23, 27, 7, 11]]
    print(len(feature_map))
    K = kit.KitNET(x_train.shape[1],maxAE,0,0, feature_map=feature_map)
    print("Running KitNET:")
    # Here we process (train/execute) each individual observation.
    # In this way, X is essentially a stream, and each observation is discarded after performing process() method.


    gsa = np.zeros([x_train.shape[0], len(feature_map)]) # a place to save the scores
    for epo in range(1):
        for i in range(x_train.shape[0]):
            if i % 1000 == 0:
                print(epo, i)
            gsa[i] = K.train(x_train[i,]) #will train during the grace periods, then execute on all the rest.
        pd.DataFrame(gsa).to_csv("gsa.csv", index=False)

    print("Running fanogan:")
    gsa = torch.from_numpy(gsa).float()
    print(gsa)
    print(gsa.shape)
    mean = gsa.mean(axis=0)  # Mean of each feature
    std = gsa.std(axis=0)
    normalize = NormalizeTransform(mean, std)
    train_mnist = SimpleDataset(gsa, y_train,transform=normalize)
    train_dataloader = DataLoader(train_mnist, batch_size=opt.batch_size,shuffle=False)
    gan_input_dim = gsa.shape[1]
    latent_dim = int(gan_input_dim * 0.5)

    generator = Generator(gan_input_dim, latent_dim)
    discriminator = Discriminator(gan_input_dim)
    train_wgangp(opt, generator, discriminator, train_dataloader, device, latent_dim)

    encoder = Encoder(gan_input_dim, latent_dim)
    train_encoder_izif(opt, generator, discriminator, encoder, train_dataloader, device)


    gsa = np.zeros([x_test.shape[0], len(feature_map)])
    for i in range(x_test.shape[0]):
        if i % 1000 == 0:
            print(i)
        gsa[i] = K.execute(x_test[i,]) #will train during the grace periods, then execute on all the rest.
    gsa = torch.from_numpy(gsa).float()
    y_test = torch.from_numpy(y_test)
    # mean = gsa.mean(axis=0)  # Mean of each feature
    # std = gsa.std(axis=0)
    # normalize = NormalizeTransform(mean, std)
    test_mnist = SimpleDataset(gsa, y_test,transform=normalize)
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
    parser.add_argument("--n_epochs", type=int, default=30,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.99,
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
