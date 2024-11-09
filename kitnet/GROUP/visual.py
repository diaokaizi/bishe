import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler    
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, classification_report
import torch
def visual(name, labels, anomaly_score):
    print("XXXXXXXXXXXXX")
    print(name)
    ########################################
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
    print("precision_recall_curve")
    print(classification_report(labels, predicted_labels))
    ########################################

    fpr, tpr, _ = roc_curve(labels, anomaly_score)
    precision, recall, _ = precision_recall_curve(labels, anomaly_score)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
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
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    labels = y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values
    anomaly_score = pd.read_csv(path)["remse"].values
    visual(name, labels, anomaly_score)

def txt(path, name):
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    labels = y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values
    anomaly_score = np.loadtxt(path)
    visual(name, labels, anomaly_score)

def load_group_np():
    seq_len=5
    embs_path = "/root/GCN/DyGCN/data/data/ugr16/graph_embs333.pt"
    # embs_path='data/graph_embs.pt'
    labels_path ="/root/GCN/DyGCN/data/data/ugr16/labels.npy"

    train_len=[0, 500]

    data_embs = torch.load(embs_path).detach().cpu().numpy()
    print(len(data_embs))
    print(data_embs.shape)
    print(data_embs[train_len[0]+seq_len:train_len[1]].shape)
    print(np.concatenate([data_embs[:train_len[0]],data_embs[train_len[1]:]]).shape)
    labels = np.load(labels_path, allow_pickle=True)
    labels=labels[seq_len:]
    labels=np.concatenate((labels[:train_len[0]], labels[train_len[1]:]))
    print(labels)
    x_train = data_embs[train_len[0]+seq_len:train_len[1]]
    test_embs=np.concatenate([data_embs[:train_len[0]],data_embs[train_len[1]:]])
    x_test = test_embs
    y_test = labels
    y_train = np.zeros(len(x_train))
    return (x_train, y_train), (x_test, y_test)

(train, y_train), (test, label) = load_group_np()


anomaly_score = pd.read_csv("RMSEs.csv")["remse"].values

plt.clf()
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC-AUC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")


visual("kitnet", label, anomaly_score)

plt.legend()
plt.savefig(f"ROC-AUC.png")
