import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 假设你的数据集格式如下
# X 为数据的特征，y 为标签
# X = ... # 你的134维特征数据
# y = ... # 对应的标签数据

# 1. 数据标准化
scaler = StandardScaler()
X = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv").drop(columns=['Row'], axis=1).values
X_scaled = scaler.fit_transform(X)

# 2. 使用 t-SNE 降维至2维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 3. 可视化 t-SNE 降维结果
plt.figure(figsize=(10, 8))
# 假设 y 中 0 为正常样本，1-4 为异常样本
y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
y_test['attack'] = y_test.idxmax(axis=1)
y_test['attack'] = y_test.apply(lambda row: "normal" if row[:5].sum() == 0 else row[-1], axis=1)
y_test['label'] = y_test.apply(lambda row: 0 if row[:5].sum() == 0 else 1, axis=1)

y = y_test['attack'].values
# 定义颜色用于标识不同标签
colors = ['blue', 'red', 'green', 'purple', 'orange']  # 用不同颜色区分五个标签


plt.scatter(X_tsne[y == "normal", 0], X_tsne[y == "normal", 1],
            label="normal", color="orange")
plt.scatter(X_tsne[y == "labelnerisbotnet", 0], X_tsne[y == "labelnerisbotnet", 1],
            label="nerisbotnet", color="blue")
plt.scatter(X_tsne[y == "labeldos", 0], X_tsne[y == "labeldos", 1],
            label="dos", color="red")
plt.scatter(X_tsne[y == "labelscan44", 0], X_tsne[y == "labelscan44", 1],
            label="scan44", color="green")
plt.scatter(X_tsne[y == "labelscan11", 0], X_tsne[y == "labelscan11", 1],
            label="scan11", color="purple")

plt.legend()
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.savefig("ok2.png")


# y_test = pd.read_csv("/root/KitNET-py/UGR16/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
# y_test['attack'] = y_test.idxmax(axis=1)
# y_test['attack'] = y_test.apply(lambda row: "normal" if row[:5].sum() == 0 else row[-1], axis=1)
# y_test['label'] = y_test.apply(lambda row: 0 if row[:5].sum() == 0 else 1, axis=1)

# print(y_test['label'].value_counts())
# print(y_test['attack'].value_counts())