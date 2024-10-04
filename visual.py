import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, classification_report
# df = pd.read_csv("results/score.csv")


# trainig_label = 0
# labels = np.where(df["label"].values == trainig_label, 0, 1)

# anomaly_score = df["anomaly_score"].values
# img_distance = df["img_distance"].values
# z_distance = df["z_distance"].values
# img_distance = anomaly_score


def visual(name, labels, anomaly_score):
    ########################################
    # # 计算 ROC 曲线
    # fpr, tpr, thresholds = roc_curve(labels, img_distance)
    # # 找到最优阈值
    # optimal_idx = np.argmax(tpr - fpr)
    # optimal_threshold = thresholds[optimal_idx] + 0.005
    # print(optimal_threshold)

    # # 使用最优阈值来生成预测标签
    # predicted_labels = np.where(img_distance >= optimal_threshold, 1, 0)
    print("XXXXXXXXXXXXX")
    print(name)

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
    plt.plot(fpr, tpr, label=f"{name} = {roc_auc:3f}")




def fgan(path, name):
    df = pd.read_csv(path)
    trainig_label = 0
    labels = np.where(df["label"].values == trainig_label, 0, 1)
    anomaly_score = df["anomaly_score"].values
    visual(name, labels, anomaly_score)

def fgan1(path, name):
    df = pd.read_csv(path)
    trainig_label = 0
    labels = np.where(df["label"].values == trainig_label, 0, 1)
    anomaly_score = df["anomaly_score"].values
    random_factors = np.random.uniform(0.1, 1.2, size=anomaly_score.shape)
    visual(name, labels, anomaly_score * random_factors)

def kitnet(path, name):
    y_test = pd.read_csv("/root/UGR16_FeatureData/csv/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    labels = y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values
    anomaly_score = pd.read_csv(path)["remse"].values
    visual(name, labels, anomaly_score)

def txt(path, name):
    y_test = pd.read_csv("/root/UGR16_FeatureData/csv/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    labels = y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values
    anomaly_score = np.loadtxt(path)
    visual(name, labels, anomaly_score)



plt.clf()
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC-AUC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

fgan("/root/urg16_compare/CAE-AnoGAN.csv", "CAE-AnoGAN")
fgan("/root/urg16_compare/f-anogan.csv", "f-anogan")
kitnet("kitnet.csv", "kitnet")
txt("ae.txt", "ae")
txt("ocsvm.txt", "ocsvm")
plt.legend()
plt.savefig(f"ROC-AUC.png")