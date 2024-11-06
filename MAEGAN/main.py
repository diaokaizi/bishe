import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from fanogan_muti.train_wgangp import train_wgangp
from fanogan_muti.train_encoder_izif import train_encoder_izif
from model.gan import Generator, Discriminator, Encoder
from tools import SimpleDataset, load_UGR16, NormalizeTransform, report_result
import model.MAE as mae
import numpy as np
import pandas as pd
from fanogan_muti.test_anomaly_detection import test_anomaly_detection
from model.MAEGAN import MAEGAN


from sklearn.preprocessing import StandardScaler, MinMaxScaler
def load_UNSW(cn = 1000, ratio = 0.12):
    # 先进行标准化
    
    # 加载训练数据
    train = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_train_1s.csv")
    train = train[~((train['binary_label_normal'] + train['binary_label_attack']) < cn)]
    # 计算每个样本的 anomaly_ratio 并筛选出 anomaly_ratio < 0.15 的样本
    train['total_records'] = train['binary_label_normal'] + train['binary_label_attack']
    train['anomaly_ratio'] = train['binary_label_attack'] / train['total_records']
    train = train[train['anomaly_ratio'] < ratio]  # 只保留 anomaly_ratio < 0.15 的样本

    # 删除不需要的列
    raw_x_train = train.drop(columns=['timestamp', 'label_background', 'label_exploits', 'label_fuzzers',
                                       'label_reconnaissance', 'label_dos', 'label_analysis', 
                                       'label_backdoor', 'label_shellcode', 'label_worms', 
                                       'label_other', 'binary_label_normal', 'binary_label_attack', 
                                       'total_records', 'anomaly_ratio'], axis=1)


    # 加载测试数据
    raw_x_test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv")

    raw_x_test = raw_x_test[~((raw_x_test['binary_label_normal'] + raw_x_test['binary_label_attack']) < cn)].drop(columns=['timestamp', 
                                       'label_background', 'label_exploits', 'label_fuzzers', 
                                       'label_reconnaissance', 'label_dos', 'label_analysis', 
                                       'label_backdoor', 'label_shellcode', 'label_worms', 
                                       'label_other', 'binary_label_normal', 'binary_label_attack'], axis=1)

    # 接下来进行归一化
    minmax_scaler = MinMaxScaler()
    
    # 对标准化后的训练数据进行归一化
    x_train_normalized = minmax_scaler.fit_transform(raw_x_train.values)  # 仅在训练数据上拟合
    
    # 对标准化后的测试数据进行归一化
    x_test_normalized = minmax_scaler.transform(raw_x_test.values)  # 使用相同的缩放器进行转换
    
    # 加载并处理测试标签
    y_test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv")
    y_test = y_test[~((y_test['binary_label_normal'] + y_test['binary_label_attack']) < cn)]
    y_test['total_records'] = y_test['binary_label_normal'] + y_test['binary_label_attack']
    y_test['anomaly_ratio'] = y_test['binary_label_attack'] / y_test['total_records']
    
    # 根据 anomaly_ratio 生成测试标签
    y_test = (y_test['anomaly_ratio'] >= ratio).astype(int).to_numpy()
    
    # 假设训练数据全部为正常数据
    y_train = np.zeros(len(x_train_normalized))

    # 输出训练和测试集的形状
    print(f"Training set shape: {x_train_normalized.shape}, Labels: {np.unique(y_train)}")
    print(f"Test set shape: {x_test_normalized.shape}, Labels: {np.unique(y_test)}")
    
    return (x_train_normalized, y_train), (x_test_normalized, y_test)


def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (x_train, y_train), (x_test, y_test) = load_UNSW()

    maegan = MAEGAN(opt, input_dim = x_train.shape[1])
    print("Running KitNET:")
    maegan.train(x_train)
    score = maegan.test(x_test, y_test)
    report_result(name="MAEGAN", anomaly_score=score, labels=y_test)
    



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
    parser.add_argument("--batch_size", type=int, default=64,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.9,
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
