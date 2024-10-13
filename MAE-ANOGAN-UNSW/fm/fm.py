import KitNET as kit
import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

def load_UNSW():
    # 先进行标准化
    standard_scaler = StandardScaler()
    
    # 加载训练数据
    train = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_train_1s.csv")
    
    # 计算每个样本的 anomaly_ratio 并筛选出 anomaly_ratio < 0.15 的样本
    train['total_records'] = train['binary_label_normal'] + train['binary_label_attack']
    train['anomaly_ratio'] = train['binary_label_attack'] / train['total_records']
    train = train[train['anomaly_ratio'] < 0.11]  # 只保留 anomaly_ratio < 0.15 的样本

    # 删除不需要的列
    raw_x_train = train.drop(columns=['timestamp', 'label_background', 'label_exploits', 'label_fuzzers',
                                       'label_reconnaissance', 'label_dos', 'label_analysis', 
                                       'label_backdoor', 'label_shellcode', 'label_worms', 
                                       'label_other', 'binary_label_normal', 'binary_label_attack', 
                                       'total_records', 'anomaly_ratio'], axis=1)

    # 标准化
    x_train_standardized = standard_scaler.fit_transform(raw_x_train.values)  # 仅在训练数据上拟合

    # 加载测试数据
    raw_x_test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv").drop(columns=['timestamp', 
                                       'label_background', 'label_exploits', 'label_fuzzers', 
                                       'label_reconnaissance', 'label_dos', 'label_analysis', 
                                       'label_backdoor', 'label_shellcode', 'label_worms', 
                                       'label_other', 'binary_label_normal', 'binary_label_attack'], axis=1)

    # 对测试数据进行标准化
    x_test_standardized = standard_scaler.transform(raw_x_test.values)  # 使用相同的缩放器进行转换

    # 接下来进行归一化
    minmax_scaler = MinMaxScaler()
    
    # 对标准化后的训练数据进行归一化
    x_train_normalized = minmax_scaler.fit_transform(x_train_standardized)  # 仅在训练数据上拟合
    
    # 对标准化后的测试数据进行归一化
    x_test_normalized = minmax_scaler.transform(x_test_standardized)  # 使用相同的缩放器进行转换
    
    # 加载并处理测试标签
    y_test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv")
    y_test['total_records'] = y_test['binary_label_normal'] + y_test['binary_label_attack']
    y_test['anomaly_ratio'] = y_test['binary_label_attack'] / y_test['total_records']
    
    # 根据 anomaly_ratio 生成测试标签
    y_test = (y_test['anomaly_ratio'] >= 0.11).astype(int).to_numpy()
    
    # 假设训练数据全部为正常数据
    y_train = np.zeros(len(x_train_normalized))

    # 输出训练和测试集的形状
    print(f"Training set shape: {x_train_normalized.shape}, Labels: {np.unique(y_train)}")
    print(f"Test set shape: {x_test_normalized.shape}, Labels: {np.unique(y_test)}")
    
    return (x_train_normalized, y_train), (x_test_normalized, y_test)

print("Reading Sample dataset...")
# X = pd.read_csv("/root/KitNET-py/UGR16/UGR16v1.Xtrain.csv").values #an m-by-n dataset with m observations
train = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtrain.csv").drop(columns=['Row'], axis=1).values
test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv").drop(columns=['Row'], axis=1).values
(train, y_train), (test, y_test) = load_UNSW()
# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = int(train.shape[0] * 0.1) #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = train.shape[0] - FMgrace #the number of instances used to train the anomaly detector (ensemble itself)
print(FMgrace, ADgrace)
# Build KitNET
K = kit.KitNET(train.shape[1],maxAE,train.shape[0] - 1,ADgrace)

print("Running KitNET:")
start = time.time()
# Here we process (train/execute) each individual observation.
# In this way, X is essentially a stream, and each observation is discarded after performing process() method.
for i in range(train.shape[0]):
    if i % 1000 == 0:
        print(i)
    K.process(train[i,]) #will train during the grace periods, then execute on all the rest.

#0.15
# [[0, 3, 75, 110, 88], [0, 3, 75, 110, 87], [0, 3, 72, 75, 110], [0, 3, 71, 75, 110], [64, 0, 3, 75, 110], [0, 3, 75, 110, 63], [0, 3, 75, 110, 62], [0, 3, 75, 110, 61], [0, 3, 75, 110, 60], [0, 3, 75, 110, 59], [0, 3, 75, 110, 58], [0, 3, 39, 75, 110], [0, 3, 38, 75, 110], [0, 3, 37, 75, 110], [0, 3, 36, 75, 110], [0, 3, 35, 75, 110], [0, 34, 3, 75, 110], [0, 3, 75, 110, 31], [0, 3, 75, 110, 30], [0, 3, 75, 110, 29], [0, 3, 75, 110, 25], [0, 3, 75, 110, 23], [0, 3, 75, 110, 21], [0, 3, 75, 110, 20], [0, 3, 75, 110, 16], [0, 3, 75, 13, 110], [0, 3, 75, 12, 110], [0, 3, 75, 11, 110], [0, 3, 10, 75, 110], [0, 3, 9, 75, 110], [0, 3, 8, 75, 110], [0, 3, 7, 75, 110], [129, 135, 79, 49, 115], [132, 138, 43, 122, 127], [96, 97, 50, 82, 94], [98, 67, 130, 75, 111], [4, 108, 44, 45, 114], [130, 100, 133, 83, 24], [102, 6, 113, 51, 57], [130, 41, 105, 75, 124], [97, 133, 82, 123, 94], [101, 137, 107, 121, 126], [134, 118, 90, 91, 93], [32, 128, 40, 137, 107], [1, 4, 73, 42, 55], [74, 77, 47, 116, 26], [134, 53, 118, 90, 124], [134, 44, 118, 90, 124], [96, 130, 133, 105, 82], [130, 82, 85, 89, 92], [134, 116, 118, 90, 93], [70, 6, 19, 54, 57], [130, 67, 105, 109, 95], [130, 75, 76, 117, 123], [33, 130, 105, 75, 84], [130, 133, 105, 82, 124], [0, 3, 99, 110, 81], [130, 109, 119, 89, 124], [134, 74, 77, 116, 26], [130, 74, 75, 78, 27], [128, 137, 107, 121, 126], [130, 134, 75, 80, 116], [130, 67, 134, 75, 111], [14, 15, 17, 18, 52, 22, 28], [66, 137, 107, 121, 126], [130, 133, 105, 83, 124], [134, 118, 119, 90, 124], [130, 6, 113, 54, 57], [130, 134, 75, 78, 27], [130, 133, 134, 116, 123], [134, 116, 86, 119, 124], [130, 104, 105, 75, 124], [0, 3, 75, 110, 111], [0, 130, 3, 75, 110], [68, 5, 73, 105, 112], [130, 105, 75, 119, 124], [65, 130, 105, 109, 119], [129, 135, 105, 79, 115], [131, 103, 136, 106, 125], [130, 105, 109, 119, 124], [132, 68, 138, 122, 127], [134, 109, 116, 119, 124], [1, 4, 73, 108, 55], [1, 4, 135, 115, 55], [129, 69, 105, 109, 119, 124], [136, 108, 114, 120, 125], [1, 131, 4, 135, 106, 108, 114, 55, 56], [137, 107, 114, 121, 126], [2, 46, 48, 122, 127]]
# [[88], [87], [72], [71], [64], [63], [62], [61], [60], [59], [58], [39], [38], [37], [36], [35], [34], [31], [30], [29], [25], [23], [21], [20], [16], [13], [12], [11], [10], [9], [8], [7], [49], [43], [50], [98], [45], [24, 100], [51, 102], [41], [94, 97], [101], [91], [32, 40], [42], [47], [53], [44], [96], [85, 92], [93], [19, 70], [95], [76, 117], [33, 84], [82], [81, 99], [89], [26, 77], [74], [128], [80], [67], [15, 14, 18, 28, 52, 17, 22], [66], [83], [90, 118], [54, 113, 6, 57], [27, 78], [123, 133], [86], [104], [111], [110, 0, 3], [5, 112], [75], [65], [79], [103], [130], [68, 132, 138, 122, 127], [116, 134], [73], [115], [105, 129, 69, 109, 119, 124], [136, 120, 125], [135, 106, 56, 131, 55, 1, 4, 108, 114], [107, 137, 121, 126], [2, 46, 48]]

# 0.11
# [[88], [87], [72], [71], [64], [63], [62], [61], [60], [59], [58], [39], [38], [37], [36], [35], [34], [31], [30], [29], [25], [23], [21], [20], [16], [13], [12], [11], [10], [9], [8], [7], [49], [43], [50], [98], [45], [24, 100], [41], [51, 102], [94, 97], [101], [91], [32, 40], [42], [47], [53], [44], [96], [85, 92], [26, 77], [93], [95], [76, 117], [33, 84], [82], [89], [19, 70], [81, 99], [128], [74], [80], [86], [67], [83], [66], [15], [14, 18, 28, 52, 17, 22], [90, 118], [27, 78], [54, 113, 6, 57], [123, 133], [104], [111], [110, 0, 3], [5, 112], [75], [65], [79], [116, 134], [130], [103], [73], [115], [105, 129, 69, 109, 119, 124], [136, 120, 125], [135, 106, 56, 131, 55, 1, 4, 108, 114], [68, 132, 138, 122, 127], [107, 137, 121, 126], [2, 46, 48]]
# [[0, 3, 75, 110, 88], [0, 3, 75, 110, 87], [0, 3, 72, 75, 110], [0, 3, 71, 75, 110], [64, 0, 3, 75, 110], [0, 3, 75, 110, 63], [0, 3, 75, 110, 62], [0, 3, 75, 110, 61], [0, 3, 75, 110, 60], [0, 3, 75, 110, 59], [0, 3, 75, 110, 58], [0, 3, 39, 75, 110], [0, 3, 38, 75, 110], [0, 3, 37, 75, 110], [0, 3, 36, 75, 110], [0, 3, 35, 75, 110], [0, 34, 3, 75, 110], [0, 3, 75, 110, 31], [0, 3, 75, 110, 30], [0, 3, 75, 110, 29], [0, 3, 75, 110, 25], [0, 3, 75, 110, 23], [0, 3, 75, 110, 21], [0, 3, 75, 110, 20], [0, 3, 75, 110, 16], [0, 3, 75, 13, 110], [0, 3, 75, 12, 110], [0, 3, 75, 11, 110], [0, 3, 10, 75, 110], [0, 3, 9, 75, 110], [0, 3, 8, 75, 110], [0, 3, 7, 75, 110], [129, 135, 79, 49, 115], [132, 138, 43, 122, 127], [133, 134, 50, 116, 123], [98, 67, 130, 75, 111], [1, 4, 44, 45, 55], [130, 100, 133, 83, 24], [130, 41, 105, 75, 124], [102, 74, 47, 51, 26], [97, 130, 133, 82, 94], [101, 137, 107, 121, 126], [134, 118, 90, 91, 93], [32, 128, 40, 137, 107], [1, 4, 42, 108, 114], [74, 77, 47, 116, 26], [134, 116, 53, 118, 90], [134, 44, 116, 118, 90], [96, 130, 133, 105, 82], [130, 82, 85, 89, 92], [134, 77, 116, 26, 124], [134, 116, 118, 90, 93], [130, 67, 105, 109, 95], [130, 75, 76, 117, 123], [33, 130, 105, 75, 84], [130, 133, 105, 82, 124], [130, 134, 116, 89, 124], [70, 6, 19, 54, 57], [99, 134, 81, 116, 124], [128, 132, 68, 105, 73], [130, 74, 75, 78, 27], [130, 134, 75, 80, 116], [134, 116, 86, 119, 124], [130, 67, 75, 111, 116], [130, 133, 105, 83, 124], [66, 137, 107, 121, 126], [14, 15, 17, 22, 28], [14, 17, 18, 52, 22, 28], [134, 116, 118, 90, 124], [130, 75, 78, 116, 27], [130, 6, 113, 54, 57], [130, 133, 134, 116, 123], [130, 134, 104, 75, 116], [0, 3, 75, 110, 111], [0, 130, 3, 75, 110], [1, 5, 73, 105, 112], [130, 105, 75, 119, 124], [65, 130, 105, 109, 119], [129, 105, 109, 79, 115], [134, 109, 116, 119, 124], [130, 105, 109, 119, 124], [131, 103, 136, 106, 125], [1, 4, 73, 108, 55], [129, 69, 109, 115, 119], [129, 69, 105, 109, 119, 124], [136, 108, 114, 120, 125], [1, 131, 4, 135, 106, 108, 114, 55, 56], [132, 68, 138, 122, 127], [137, 107, 114, 121, 126], [2, 46, 48, 122, 127]]