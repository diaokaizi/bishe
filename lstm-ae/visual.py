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
    f1_scores = 2 * (precision * recall) / (precision + recall)
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
    y_test = torch.from_numpy(y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values)
    return (x_train, y_train), (x_test, y_test)
