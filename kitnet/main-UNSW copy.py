import sys
sys.path.append('../..')
import KitNET as kit
import numpy as np
import pandas as pd
import time
##############################################################################
# KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders.
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection

# This script demonstrates KitNET's ability to incrementally learn, and detect anomalies.
# The demo involves an m-by-n dataset with n=115 dimensions (features), and m=100,000 observations.
# Each observation is a snapshot of the network's state in terms of incremental damped statistics (see the NDSS paper for more details)

#The runtimes presented in the paper, are based on the C++ implimentation (roughly 100x faster than the python implimentation)
###################  Last Tested with Anaconda 2.7.14   #######################

# Load sample dataset (a recording of the Mirai botnet malware being activated)
# The first 70,000 observations are clean...


print("Reading Sample dataset...")
# X = pd.read_csv("/root/KitNET-py/UGR16/UGR16v1.Xtrain.csv").values #an m-by-n dataset with m observations
# KitNET params:

train = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_train_1s.csv")
# 计算每个样本的 anomaly_ratio 并筛选出 anomaly_ratio < 0.15 的样本
train['total_records'] = train['binary_label_normal'] + train['binary_label_attack']
train['anomaly_ratio'] = train['binary_label_attack'] / train['total_records']
print(train[train['anomaly_ratio'] < 0.08].shape)
print(train[train['anomaly_ratio'] < 0.09].shape)
print(train[train['anomaly_ratio'] < 0.10].shape)
print(train[train['anomaly_ratio'] < 0.11].shape)
print(train[train['anomaly_ratio'] < 0.12].shape)

train = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv")
# 计算每个样本的 anomaly_ratio 并筛选出 anomaly_ratio < 0.15 的样本
train['total_records'] = train['binary_label_normal'] + train['binary_label_attack']
train['anomaly_ratio'] = train['binary_label_attack'] / train['total_records']
print(train[train['anomaly_ratio'] < 0.08].shape)
print(train[train['anomaly_ratio'] < 0.09].shape)
print(train[train['anomaly_ratio'] < 0.10].shape)
print(train[train['anomaly_ratio'] < 0.11].shape)
print(train[train['anomaly_ratio'] < 0.12].shape)
print(train[train['anomaly_ratio'] < 0.13].shape)
print(train[train['anomaly_ratio'] < 0.14].shape)