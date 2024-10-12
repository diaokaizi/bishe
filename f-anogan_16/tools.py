import torch
from torchvision import datasets
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class NormalizeTransform:
    """ Normalize features with mean and standard deviation. """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std + 1e-8

    def __call__(self, sample):
        return (sample - self.mean) / self.std

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]

        if self.transform:
            data = self.transform(data)

        return data, labels

    def __len__(self):
        return len(self.data)

def fix_name():
    return ["sportsocks", "sportprivate", "dportirc", "sporttelnet", "sportrapservice", "dporthttp",
            "sportsyslog", "sportreserved", "dportkpasswd", "tcpflagsACK", "npacketsmedium",
            "sportcups", "dporttelnet", "sportldaps", "tcpflagsPSH", "dportoracle"]

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
    return (x_train, y_train), (x_test, y_test)

def load_UGR16_16():
    raw_x_train = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)
    x_train = raw_x_train[fix_name()]
    x_train = torch.from_numpy(x_train.values).float()
    y_train = torch.zeros(len(x_train))


    raw_x_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
    x_test = raw_x_test[fix_name()]
    x_test = torch.from_numpy(x_test.values).float()
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    y_test = torch.from_numpy(y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values)
    return (x_train, y_train), (x_test, y_test)

def load_UGR16_StandardScaler():
    # 先进行标准化
    standard_scaler = StandardScaler()
    
    # 加载并预处理训练数据
    raw_x_train = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)
    x_train_standardized = standard_scaler.fit_transform(raw_x_train.values)  # 仅在训练数据上拟合
    
    # 加载并预处理测试数据
    raw_x_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
    x_test_standardized = standard_scaler.transform(raw_x_test.values)  # 使用相同的缩放器进行转换
    
    # 接下来进行归一化
    minmax_scaler = MinMaxScaler()
    
    # 对标准化后的训练数据进行归一化
    x_train_normalized = minmax_scaler.fit_transform(x_train_standardized)  # 仅在训练数据上拟合
    x_train_normalized = torch.from_numpy(x_train_normalized).float()
    
    # 对标准化后的测试数据进行归一化
    x_test_normalized = minmax_scaler.transform(x_test_standardized)  # 使用相同的缩放器进行转换
    x_test_normalized = torch.from_numpy(x_test_normalized).float()
    
    # 加载并处理测试标签
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(
        columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], 
        axis=1
    )
    y_test = y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1)
    print(y_test.value_counts())
    y_test = torch.from_numpy(y_test.values)
    
    y_train = torch.zeros(len(x_train_normalized))  # 假设训练数据全部为正常数据
    
    return (x_train_normalized, y_train), (x_test_normalized, y_test)

def load_UGR16_gas():
    raw_x_train = pd.read_csv("/root/GSA-AnoGAN/KitNet/gsa.csv", header=None).drop(columns=[0], axis=1)
    x_train = torch.from_numpy(raw_x_train.values).float()
    y_train = torch.zeros(len(x_train))
    return (x_train, y_train), (0, 0)
