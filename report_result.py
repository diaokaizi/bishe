import torch
from torchvision import datasets
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import os
import csv

def report_result(model, name, labels, anomaly_score, fun="pr"):
    base_path = os.path.join("results", name, model)
    print(base_path)
    os.makedirs(base_path, exist_ok=True)
    with open(os.path.join(base_path, "labels_anomaly_scores.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["label", "anomaly_score"])  # 写入表头
        writer.writerows(zip(labels, anomaly_score))  # 写入数据行
    print("XXXXXXXXXXXXX")
    print(name)
    if fun == "roc":
        fpr, tpr, thresholds = roc_curve(labels, anomaly_score)
        # 找到最优阈值
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
    else:
        precision, recall, thresholds = precision_recall_curve(labels, anomaly_score)
        f1_scores = np.where((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
    # 根据最优阈值生成预测标签
    predicted_labels = np.where(anomaly_score >= optimal_threshold, 1, 0)
    print("precision_recall_curve")
    # print(classification_report(labels, predicted_labels))
    print(classification_report(labels, predicted_labels))
    report_dict = classification_report(labels, predicted_labels, output_dict=True)
    # 手动格式化输出，保留4位小数
    for label, metrics in report_dict.items():
        if label == "1":
            if isinstance(metrics, dict):  # 这是每个标签的指标部分
                print(f"Label: {label}")
                for metric, score in metrics.items():
                    print(f"  {metric}: {score:.4f}")
            else:  # 这是整体的指标部分，如 'accuracy'
                print(f"{label}: {metrics:.4f}")
    fpr, tpr, _ = roc_curve(labels, anomaly_score)
    precision, recall, _ = precision_recall_curve(labels, anomaly_score)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.clf()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(fpr, tpr, label=f"{name} = {roc_auc:3f}")
    plt.title("ROC-AUC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(base_path, "ROC-AUC.png"))

    t, auc_score=plot_roc(labels, anomaly_score)
    print("auc_score ", auc_score)
    labels = np.array(labels)
    anomaly_score = np.array(anomaly_score)
    pred=np.ones(len(anomaly_score))
    pred[anomaly_score<t]=0
    print("acc:{:.4f},pre{:.4f},rec:{:.4f}, f1:{:.4f}".format(accuracy_score(labels, pred),precision_score(labels, pred),recall_score(labels, pred), f1_score(labels, pred)))
    with open(os.path.join(base_path, "result.txt"), "w") as f:
        f.write(f"auc_score: {auc_score}\n")
        f.write("acc:{:.4f},pre{:.4f},rec:{:.4f}, f1:{:.4f}".format(accuracy_score(labels, pred),precision_score(labels, pred),recall_score(labels, pred), f1_score(labels, pred)))
    
    return f1_score(labels, pred)

def plot_roc(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
    threshold = thresholds[maxindex]
    print('异常阈值', threshold)
    auc_score = auc(fpr, tpr)
    print('auc值: {:.4f}'.format(auc_score))
    return threshold, auc_score
