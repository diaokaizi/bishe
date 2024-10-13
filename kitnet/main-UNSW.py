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
K = kit.KitNET(train.shape[1],maxAE,train.shape[0],train.shape[0])

print("Running KitNET:")
start = time.time()
# Here we process (train/execute) each individual observation.
# In this way, X is essentially a stream, and each observation is discarded after performing process() method.
for i in range(train.shape[0]):
    if i % 1000 == 0:
        print(i)
    K.process(train[i,]) #will train during the grace periods, then execute on all the rest.
for i in range(train.shape[0]):
    if i % 1000 == 0:
        print(i)
    K.process(train[i,]) #will train during the grace periods, then execute on all the rest.

RMSEs = np.zeros(test.shape[0]) # a place to save the scores
for i in range(test.shape[0]):
    if i % 1000 == 0:
        print(i)
    RMSEs[i] = K.process(test[i,]) #will train during the grace periods, then execute on all the rest.
pd.DataFrame(RMSEs).to_csv("/root/bishe/kitnet/UNSW/RMSEs10.csv", header=["remse"])
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))