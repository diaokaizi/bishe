import numpy as np
import pandas as pd
attack = pd.read_csv("/root/work/mymode/attack.csv", header=None)
normal = pd.read_csv("/root/work/mymode/normal.csv", header=None)
normal = normal[normal[12] == "background"]
data = pd.concat([attack, normal])
data = data.sample(n=500000)
print(data[12].value_counts())

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# 假设你的数据集格式如下
# X 为数据的特征，y 为标签
# X = ... # 你的134维特征数据
# y = ... # 对应的标签数据

# 1. 数据标准化
scaler = StandardScaler()
# 初始化 LabelEncoder
label_encoder = LabelEncoder()

# 将 'Category' 列的字符串转换为数字
data[6] = label_encoder.fit_transform(data[6])
data[7] = label_encoder.fit_transform(data[7])
print(data)


X = data.drop(columns=[0, 2, 3, 12], axis=1).values
X_scaled = scaler.fit_transform(X)

# 2. 使用 t-SNE 降维至2维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 3. 可视化 t-SNE 降维结果
plt.figure(figsize=(10, 8))

# 假设 y 中 0 为正常样本，1-4 为异常样本
y = data[12].values
# 定义颜色用于标识不同标签
colors = ['blue', 'red', 'green', 'purple', 'orange']  # 用不同颜色区分五个标签

plt.scatter(X_tsne[y == "background", 0], X_tsne[y == "background", 1],
            label="normal", color="orange")
plt.scatter(X_tsne[y == "nerisbotnet", 0], X_tsne[y == "nerisbotnet", 1],
            label="nerisbotnet", color="blue")
plt.scatter(X_tsne[y == "dos", 0], X_tsne[y == "dos", 1],
            label="dos", color="red")
plt.scatter(X_tsne[y == "scan44", 0], X_tsne[y == "scan44", 1],
            label="scan44", color="green")
plt.scatter(X_tsne[y == "scan11", 0], X_tsne[y == "scan11", 1],
            label="scan11", color="purple")
    


plt.legend()
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.savefig("ok2.png")