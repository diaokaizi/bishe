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
def load_UNSW(ratio = 0.11):
    # 加载并处理测试标签
    y_test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv")
    y_test['total_records'] = y_test['binary_label_normal'] + y_test['binary_label_attack']
    y_test['anomaly_ratio'] = y_test['binary_label_attack'] / y_test['total_records']
    
    # 根据 anomaly_ratio 生成测试标签
    y_test = (y_test['anomaly_ratio'] >= ratio).astype(int).to_numpy()
    return y_test




def visual(name, labels, anomaly_score, fun="pr"):
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
    print(classification_report(labels, predicted_labels))
    fpr, tpr, _ = roc_curve(labels, anomaly_score)
    precision, recall, _ = precision_recall_curve(labels, anomaly_score)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.plot(fpr, tpr, label=f"{name} = {roc_auc:3f}")

def fgan(path, name):
    df = pd.read_csv(path)
    anomaly_score = df["anomaly_score"].values
    y_test = load_UNSW()
    visual(name, y_test, anomaly_score)

def kitnet(path, name):
    anomaly_score = pd.read_csv(path)["remse"].values
    y_test = load_UNSW(0.12)
    visual(name, y_test, anomaly_score)

def txt(path, name, fun="pr"):
    anomaly_score = np.loadtxt(path)
    y_test = load_UNSW()
    visual(name, y_test, anomaly_score, fun)

def maegan(path, name):
    # tt = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_train_1s.csv")
    # tt['total_records'] = tt['binary_label_normal'] + tt['binary_label_attack']
    # tt['anomaly_ratio'] = tt['binary_label_attack'] / tt['total_records']
    # tt = tt[~((tt['binary_label_normal'] + tt['binary_label_attack']) < 1000)]
    # tt = (tt['anomaly_ratio'] >= 0.11).astype(int).to_numpy()
    # print(len(tt))
    df = pd.read_csv(path)
    label = df["label"].values
    anomaly_score = df["anomaly_score"].values
    visual(name, label, anomaly_score)


plt.clf()
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC-AUC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
txt("/root/bishe/visual-UNSW/DeepSVDD_config1_run1.txt", "DeepSVDD", "roc")
txt("/root/bishe/visual-UNSW/SLAD_config12_run1.txt", "SLAD")
txt("/root/bishe/visual-UNSW/RCA_config8_run1.txt", "RCA", "roc")
# fgan("/root/bishe/MAE-ANOGAN-UNSW/results/score-ok.csv", "ae")
# fgan("/root/bishe/f-anogan-UNSW/results/score-11.csv", "ocsvm")
# # txt("/root/bishe/ocsvm/UNSW_best_params_fin.txt", "ocsvm")
# txt("/root/bishe/iForest/UNSW_best_params-11.txt", "iForest")
# fgan("/root/bishe/f-anogan-UNSW/results/score.csv", "f-anogan")
# kitnet("/root/bishe/kitnet/UNSW/RMSEs10.csv", "kitnet")
# maegan("/root/bishe/MAE-ANOGAN-UNSW/results/score.csv", "MAE-ANOGAN")

# txt("DeepSVDD_config6_run1.txt", "DeepSVDD")
# txt("SLAD_config12_run1_fake.txt", "SLAD")


# kitnet("/root/bishe/kitnet-fm/RMSEs_raw.csv", "RMSEs_raw")
# kitnet("/root/bishe/kitnet-fm/RMSEs10.csv", "RMSEs10")
# kitnet("/root/bishe/kitnet-fm/RMSEs.csv", "kitnet")
# kitnet("/root/bishe/kitnet-fm/RMSEs copy.csv", "copy")



plt.legend()
plt.savefig(f"ROC-AUC.png")