import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, classification_report
 
# labels = y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values

def visual(name, labels, anomaly_score):
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    print("XXXXXXXXXXXXX")
    print(name)
    ########################################
    # # # 计算 ROC 曲线
    # fpr, tpr, thresholds = roc_curve(labels, anomaly_score)
    # # 找到最优阈值
    # optimal_idx = np.argmax(tpr - fpr)
    # optimal_threshold = thresholds[optimal_idx] + 0.005
    # print(optimal_threshold)

    # # 使用最优阈值来生成预测标签
    # predicted_labels = np.where(anomaly_score >= optimal_threshold, 1, 0)
    # print("roc_curve")
    # print(classification_report(labels, predicted_labels))

    precision, recall, thresholds = precision_recall_curve(labels, anomaly_score)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(optimal_threshold)

    # 根据最优阈值生成预测标签
    predicted_labels = np.where(anomaly_score >= optimal_threshold, 1, 0)
    recall_per_type = {}
    anomaly_columns =['labeldos', 'labelscan11', 'labelscan44', 'labelnerisbotnet']
    for col in anomaly_columns:
        true_labels = y_test[col]  # 当前异常类型的真实标签
        # 计算当真实标签为1时，预测也为1的比例（单独计算每列的精确率）
        binary_labels = (true_labels > 0).astype(int)
        recall = recall_score(binary_labels, predicted_labels, average=None)
        recall_per_type[col] = recall

    # 打印每种异常类型的准确率
    for anomaly_type, recall in recall_per_type.items():
        print(f"Precision for {anomaly_type}: {recall[1]:.4f}")

def kitnet(path, name):
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist", "labelanomalyspam"], axis=1)
    labels = y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values
    anomaly_score = pd.read_csv(path)["remse"].values
    visual(name, labels, anomaly_score)

def fgan(path, name):
    df = pd.read_csv(path)
    trainig_label = 0
    labels = np.where(df["label"].values == trainig_label, 0, 1)
    anomaly_score = df["anomaly_score"].values
    visual(name, labels, anomaly_score)


fgan("f-gan.csv", "MAE-ANOGAN")
kitnet("/root/bishe/kitnet/RMSEs.csv", "kitnet")
fgan("/root/bishe/MAE-ANOGAN2/results/score.csv", "MAE-ANOGAN2")
fgan("/root/bishe/MAE-ANOGAN/results/score.csv", "MAE-ANOGAN1")

# y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
# true_labels = y_test["labeldos"]
# binary_labels = (true_labels > 0).astype(int)
# print(binary_labels.value_counts())
# true_labels = y_test["labelscan11"]
# binary_labels = (true_labels > 0).astype(int)
# print(binary_labels.value_counts())
# true_labels = y_test["labelscan44"]
# binary_labels = (true_labels > 0).astype(int)
# print(binary_labels.value_counts())
# true_labels = y_test["labelnerisbotnet"]
# binary_labels = (true_labels > 0).astype(int)
# print(binary_labels.value_counts())