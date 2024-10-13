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
print(train.shape)
# 计算每个样本的 anomaly_ratio 并筛选出 anomaly_ratio < 0.15 的样本
train['total_records'] = train['binary_label_normal'] + train['binary_label_attack']
train['anomaly_ratio'] = train['binary_label_attack'] / train['total_records']
train = train[train['anomaly_ratio'] < 0.11]

# 删除不需要的列，包括 total_records 和 anomaly_ratio
train = train.drop(columns=['timestamp', 'label_background','label_exploits','label_fuzzers','label_reconnaissance',
                            'label_dos','label_analysis','label_backdoor','label_shellcode','label_worms','label_other',
                            'binary_label_normal','binary_label_attack', 'total_records', 'anomaly_ratio'], axis=1)

# 将筛选后的 DataFrame 转为 NumPy 数组
train = train.values
print(train.shape)
test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv").drop(columns=['timestamp', 'label_background','label_exploits','label_fuzzers','label_reconnaissance','label_dos','label_analysis','label_backdoor','label_shellcode','label_worms','label_other','binary_label_normal','binary_label_attack'], axis=1).values #an m-by-n dataset with m observations
print(test.shape)

maxAE = 10 #maximum size for any autoencoder in the ensemble layer
# Build KitNET
feature_map = [[0, 3, 75, 110, 88], [0, 3, 75, 110, 87], [0, 3, 72, 75, 110], [0, 3, 71, 75, 110], [64, 0, 3, 75, 110], [0, 3, 75, 110, 63], [0, 3, 75, 110, 62], [0, 3, 75, 110, 61], [0, 3, 75, 110, 60], [0, 3, 75, 110, 59], [0, 3, 75, 110, 58], [0, 3, 39, 75, 110], [0, 3, 38, 75, 110], [0, 3, 37, 75, 110], [0, 3, 36, 75, 110], [0, 3, 35, 75, 110], [0, 34, 3, 75, 110], [0, 3, 75, 110, 31], [0, 3, 75, 110, 30], [0, 3, 75, 110, 29], [0, 3, 75, 110, 25], [0, 3, 75, 110, 23], [0, 3, 75, 110, 21], [0, 3, 75, 110, 20], [0, 3, 75, 110, 16], [0, 3, 75, 13, 110], [0, 3, 75, 12, 110], [0, 3, 75, 11, 110], [0, 3, 10, 75, 110], [0, 3, 9, 75, 110], [0, 3, 8, 75, 110], [0, 3, 7, 75, 110], [129, 135, 79, 49, 115], [132, 138, 43, 122, 127], [133, 134, 50, 116, 123], [98, 67, 130, 75, 111], [1, 4, 44, 45, 55], [130, 100, 133, 83, 24], [130, 41, 105, 75, 124], [102, 74, 47, 51, 26], [97, 130, 133, 82, 94], [101, 137, 107, 121, 126], [134, 118, 90, 91, 93], [32, 128, 40, 137, 107], [1, 4, 42, 108, 114], [74, 77, 47, 116, 26], [134, 116, 53, 118, 90], [134, 44, 116, 118, 90], [96, 130, 133, 105, 82], [130, 82, 85, 89, 92], [134, 77, 116, 26, 124], [134, 116, 118, 90, 93], [130, 67, 105, 109, 95], [130, 75, 76, 117, 123], [33, 130, 105, 75, 84], [130, 133, 105, 82, 124], [130, 134, 116, 89, 124], [70, 6, 19, 54, 57], [99, 134, 81, 116, 124], [128, 132, 68, 105, 73], [130, 74, 75, 78, 27], [130, 134, 75, 80, 116], [134, 116, 86, 119, 124], [130, 67, 75, 111, 116], [130, 133, 105, 83, 124], [66, 137, 107, 121, 126], [14, 15, 17, 22, 28], [14, 17, 18, 52, 22, 28], [134, 116, 118, 90, 124], [130, 75, 78, 116, 27], [130, 6, 113, 54, 57], [130, 133, 134, 116, 123], [130, 134, 104, 75, 116], [0, 3, 75, 110, 111], [0, 130, 3, 75, 110], [1, 5, 73, 105, 112], [130, 105, 75, 119, 124], [65, 130, 105, 109, 119], [129, 105, 109, 79, 115], [134, 109, 116, 119, 124], [130, 105, 109, 119, 124], [131, 103, 136, 106, 125], [1, 4, 73, 108, 55], [129, 69, 109, 115, 119], [129, 69, 105, 109, 119, 124], [136, 108, 114, 120, 125], [1, 131, 4, 135, 106, 108, 114, 55, 56], [132, 68, 138, 122, 127], [137, 107, 114, 121, 126], [2, 46, 48, 122, 127]]
K = kit.KitNET(train.shape[1],maxAE,0,0, feature_map=feature_map)

print("Running KitNET:")
start = time.time()
# Here we process (train/execute) each individual observation.
# In this way, X is essentially a stream, and each observation is discarded after performing process() method.
for t in range(1):
    for i in range(train.shape[0]):
        loss = K.train(train[i,]) #will train during the grace periods, then execute on all the rest.
        if i % 1000 == 0:
            print(t, i, loss)

RMSEs = np.zeros(test.shape[0]) # a place to save the scores

for i in range(test.shape[0]):
    if i % 1000 == 0:
        print(i)
    RMSEs[i] = K.execute(test[i,])
pd.DataFrame(RMSEs).to_csv("RMSEs-UNSW-11.csv", header=["remse"])
stop = time.time()