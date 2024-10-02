import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler    
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, classification_report
test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_use.csv")
labels = test['binary_label_attack'].apply(lambda x: 0 if x == 0 else 1).values

img_distance = pd.read_csv("/root/bishe/kitnet/result/RMSEs.csv")["score"].values


########################################
# # 计算 ROC 曲线
# fpr, tpr, thresholds = roc_curve(labels, img_distance)
# # 找到最优阈值
# optimal_idx = np.argmax(tpr - fpr)
# optimal_threshold = thresholds[optimal_idx]
# print(optimal_threshold)

# # 使用最优阈值来生成预测标签
# predicted_labels = np.where(img_distance >= optimal_threshold, 1, 0)

precision, recall, thresholds = precision_recall_curve(labels, img_distance)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(optimal_threshold)

# 根据最优阈值生成预测标签
predicted_labels = np.where(img_distance >= optimal_threshold, 1, 0)

print('\tUSING ROC-CURVE & Youden:\n')
print(f'\t\tAccuracy={accuracy_score(labels, predicted_labels)}\n\t\tPrecision={precision_score(labels, predicted_labels)}\n\t\tRecall={recall_score(labels, predicted_labels)}\n\t\tF1={f1_score(labels, predicted_labels)}\n')
print('\tUSING PR-CURVE & Distance:\n')
print(f'\t\tAccuracy={accuracy_score(labels, predicted_labels)}\n\t\tPrecision={precision_score(labels, predicted_labels)}\n\t\tRecall={recall_score(labels, predicted_labels)}\n\t\tF1={f1_score(labels, predicted_labels)}\n')
print(classification_report(labels, predicted_labels))
########################################




fpr, tpr, _ = roc_curve(labels, img_distance)
precision, recall, _ = precision_recall_curve(labels, img_distance)
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
plt.savefig("/root/bishe/kitnet/result/ROC-AUC.png")

plt.clf()
plt.plot(recall, precision, label=f"PR = {pr_auc:3f}")
plt.title("PR-AUC")
plt.xlabel("Recall")
plt.ylabel("Pecision")
plt.legend()
plt.savefig("/root/bishe/kitnet/result/PR-AUC.png")

# plt.clf()

plt.clf()
plt.hist([img_distance[labels == 0] ,[val for val in img_distance[labels == 1] for i in range(3)]],
        bins=1000, density=True, stacked=True,
        label=["Normal", "Abnormal"])
plt.title("Discrete distributions of anomaly scores")
plt.xlabel("Anomaly scores A(x)")
plt.ylabel("h")
plt.legend()
plt.savefig("Discrete distributions of anomaly scores.png")


# plt.hist([anomaly_score[labels == 0] ,[val for val in anomaly_score[labels == 1] for i in range(3)]],
#           bins=1000, density=True, stacked=True,
#           label=["Normal", "Abnormal"])
# plt.xlim(0, 0.2)
# plt.title("Discrete distributions of anomaly scores")
# plt.xlabel("Anomaly scores A(x)")
# plt.ylabel("h")
# plt.legend()
# plt.savefig("Discrete distributions of anomaly scores.png")