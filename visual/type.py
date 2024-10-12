import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, recall_score

def visual(name, labels, anomaly_score):
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    # 计算 Precision-Recall 曲线
    precision, recall, thresholds = precision_recall_curve(labels, anomaly_score)
    # 计算 F1 分数
    f1_scores = 2 * (precision * recall) / (precision + recall)
    # 找到 F1 分数最大的阈值
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold for {name}: {optimal_threshold}")

    # 根据最优阈值生成预测标签
    predicted_labels = np.where(anomaly_score >= optimal_threshold, 1, 0)
    recall_per_type = {}
    anomaly_columns = ['labeldos', 'labelscan11', 'labelscan44', 'labelnerisbotnet']
    for col in anomaly_columns:
        true_labels = y_test[col].values  # 当前异常类型的真实标签
        binary_labels = (true_labels > 0).astype(int)
        # 计算召回率
        recall = recall_score(binary_labels, predicted_labels, zero_division=0)
        recall_per_type[col] = recall

    # 打印每种异常类型的召回率
    for anomaly_type, recall in recall_per_type.items():
        print(f"Recall for {name} on {anomaly_type}: {recall:.4f}")

    # 返回模型名称和对应的召回率
    return name, recall_per_type

def kitnet(path, name):
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist", "labelanomalyspam"], axis=1)
    labels = y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values
    anomaly_score = pd.read_csv(path)["remse"].values
    return visual(name, labels, anomaly_score)

def fgan(path, name):
    df = pd.read_csv(path)
    training_label = 0
    labels = np.where(df["label"].values == training_label, 0, 1)
    anomaly_score = df["anomaly_score"].values
    return visual(name, labels, anomaly_score)

# 收集每个模型的召回率数据
results = []

# 调用各个模型的函数并收集结果
results.append(fgan("/root/bishe/f-anogan_16/results/score_UGR16_old.csv", "f-anogan"))
results.append(kitnet("/root/bishe/kitnet/RMSEs.csv", "KitNet"))
results.append(fgan("/root/bishe/MAE-ANOGAN2/results/score.csv", "MAE-ANOGAN2"))
# results.append(fgan("/root/bishe/MAE-ANOGAN/results/score.csv", "MAE-ANOGAN1"))

# 将结果整理成 DataFrame，便于绘图
models = []
recall_data = []

for name, recall_per_type in results:
    models.append(name)
    recall_data.append(recall_per_type)

df = pd.DataFrame(recall_data, index=models)

# 将列名转换为更友好的名称
df.columns = ['DoS', 'Scan11', 'Scan44', 'NerisBotnet']

# 转置 DataFrame，使异常种类为 x 轴
df = df.T

# 绘制柱状图
df.plot(kind='bar', figsize=(10, 6))
plt.title('Recall for Different Anomaly Types Across Models')
plt.xlabel('Anomaly Types')
plt.ylabel('Recall')
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("anomaly_type_recall.png")