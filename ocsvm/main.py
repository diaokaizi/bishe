import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder, MinMaxScaler
from visual import visual
from sklearn.svm import OneClassSVM
# 准备数据
np.random.seed(42)

def fix_name():
    return ["sportsocks", "sportprivate", "dportirc", "sporttelnet", "sportrapservice", "dporthttp",
            "sportsyslog", "sportreserved", "dportkpasswd", "tcpflagsACK", "npacketsmedium",
            "sportcups", "dporttelnet", "sportldaps", "tcpflagsPSH", "dportoracle"]

def load_after():
    scaler = StandardScaler()
    raw_x_train = pd.read_csv("/root/faac-compare/data/after/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)
    x_train = scaler.fit_transform(raw_x_train.values)
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.zeros(len(x_train))


    raw_x_test = pd.read_csv("/root/faac-compare/data/after/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
    x_test = scaler.fit_transform(raw_x_test.values)
    x_test = torch.from_numpy(x_test).float()
    y_test = pd.read_csv("/root/faac-compare/data/after/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    print(y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).value_counts())
    y_test = torch.from_numpy(y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values)
    return (x_train, y_train), (x_test, y_test)


name = "after"
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.02)
(x_train, y_train), (x_test, y_test) = load_after()
ocsvm.fit(x_train)
decision_scores = ocsvm.decision_function(x_test)
anomaly_scores = -decision_scores
scaler = MinMaxScaler()
anomaly_scores_normalized = scaler.fit_transform(anomaly_scores.reshape(-1, 1)).flatten()
np.savetxt(f"{name}.txt", anomaly_scores_normalized)
visual(name, y_test, anomaly_scores_normalized)