import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, classification_report
# df = pd.read_csv("results/score.csv")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder






# trainig_label = 0
# labels = np.where(df["label"].values == trainig_label, 0, 1)

# anomaly_score = df["anomaly_score"].values
# img_distance = df["img_distance"].values
# z_distance = df["z_distance"].values
# img_distance = anomaly_score


def visual(name, labels, anomaly_score):
    ########################################
    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(labels, anomaly_score)
#     找到最优阈值
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx] + 0.005
    print(optimal_threshold)

    precision, recall, thresholds = precision_recall_curve(labels, anomaly_score)
    f1_scores = np.where((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    print(optimal_threshold)

    # 根据最优阈值生成预测标签
    predicted_labels = np.where(anomaly_score >= optimal_threshold, 1, 0)

    # print('\tUSING ROC-CURVE & Youden:\n')
    # print(f'\t\tAccuracy={accuracy_score(labels, predicted_labels)}\n\t\tPrecision={precision_score(labels, predicted_labels)}\n\t\tRecall={recall_score(labels, predicted_labels)}\n\t\tF1={f1_score(labels, predicted_labels)}\n')
    # print('\tUSING PR-CURVE & Distance:\n')
    # print(f'\t\tAccuracy={accuracy_score(labels, predicted_labels)}\n\t\tPrecision={precision_score(labels, predicted_labels)}\n\t\tRecall={recall_score(labels, predicted_labels)}\n\t\tF1={f1_score(labels, predicted_labels)}\n')
    print(classification_report(labels, predicted_labels))
    ########################################

    fpr, tpr, _ = roc_curve(labels, anomaly_score)
    precision, recall, _ = precision_recall_curve(labels, anomaly_score)
    roc_auc = auc(fpr, tpr)
    pr_auc =  auc(recall, precision)

    print(roc_auc)
    print(pr_auc)
    plt.clf()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC-AUC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f"{name}-ROC-AUC.png")

    plt.clf()
    plt.plot(recall, precision, label=f"PR = {pr_auc:3f}")
    plt.title("PR-AUC")
    plt.xlabel("Recall")
    plt.ylabel("Pecision")
    plt.legend()
    plt.savefig(f"{name}-PR-AUC.png")

    plt.clf()
    plt.hist([anomaly_score[labels == 0] ,[val for val in anomaly_score[labels == 1] for i in range(3)]],
            bins=1000, density=True, stacked=True,
            label=["Normal", "Abnormal"])
    plt.title("Discrete distributions of anomaly scores")
    plt.xlabel("Anomaly scores A(x)")
    plt.ylabel("h")
    plt.legend()
    plt.savefig("Discrete distributions of anomaly scores.png")


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

def load_UNSW():
    
    # 加载训练数据
    train = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_train_1s.csv")
    
    # 计算每个样本的 anomaly_ratio 并筛选出 anomaly_ratio < 0.15 的样本
    train['total_records'] = train['binary_label_normal'] + train['binary_label_attack']
    train['anomaly_ratio'] = train['binary_label_attack'] / train['total_records']
    train = train[train['anomaly_ratio'] < 0.15]  # 只保留 anomaly_ratio < 0.15 的样本

    # 删除不需要的列
    raw_x_train = train.drop(columns=['timestamp', 'label_background', 'label_exploits', 'label_fuzzers',
                                       'label_reconnaissance', 'label_dos', 'label_analysis', 
                                       'label_backdoor', 'label_shellcode', 'label_worms', 
                                       'label_other', 'binary_label_normal', 'binary_label_attack', 
                                       'total_records', 'anomaly_ratio'], axis=1)

    # 标准化
    x_train_standardized = torch.from_numpy(raw_x_train.values).float()  # 仅在训练数据上拟合
    
    # 加载测试数据
    raw_x_test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv").drop(columns=['timestamp', 
                                       'label_background', 'label_exploits', 'label_fuzzers', 
                                       'label_reconnaissance', 'label_dos', 'label_analysis', 
                                       'label_backdoor', 'label_shellcode', 'label_worms', 
                                       'label_other', 'binary_label_normal', 'binary_label_attack'], axis=1)

    # 对测试数据进行标准化
    x_test_standardized = torch.from_numpy(raw_x_test.values).float()  # 使用相同的缩放器进行转换

    
    # 加载并处理测试标签
    y_test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv")
    y_test['total_records'] = y_test['binary_label_normal'] + y_test['binary_label_attack']
    y_test['anomaly_ratio'] = y_test['binary_label_attack'] / y_test['total_records']
    
    # 根据 anomaly_ratio 生成测试标签
    y_test = torch.from_numpy((y_test['anomaly_ratio'] > 0.15).astype(int).to_numpy())
    
    # 假设训练数据全部为正常数据
    y_train = torch.zeros(len(x_train_standardized))

    # 输出训练和测试集的形状
    print(f"Training set shape: {x_train_standardized.shape}, Labels: {y_train.unique()}")
    print(f"Test set shape: {x_train_standardized.shape}, Labels: {y_test.unique()}")
    return (x_train_standardized, y_train), (x_test_standardized, y_test)


(x_train, y_train), (x_test, y_test) = load_UNSW()
mse_losses = np.loadtxt("UNSW.txt")
visual("UNSW", y_test, mse_losses)