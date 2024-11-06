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
    (x_train, y_train), (x_test, y_test) = load_UGR16()

    maegan = MAEGAN(opt, input_dim = x_train.shape[1], filepath="ugr16")
    print("Running KitNET:")
    maegan.train(x_train)
    score = maegan.test(x_test, y_test)
    report_result(name="ugr16", anomaly_score=score, labels=y_test)
    



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
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for "
                             "discriminator per iter")
    parser.add_argument("--seed", type=int, default=42,
                        help="value of a random seed")
    opt = parser.parse_args()

    main(opt)
