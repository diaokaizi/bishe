import pandas as pd
import matplotlib.pyplot as plt

# 加载训练数据和测试数据
train = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_train_1s.csv")
y_test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv")

# 计算训练数据中的 anomaly_ratio
train['total_records'] = train['binary_label_normal'] + train['binary_label_attack']
train['anomaly_ratio'] = train['binary_label_attack'] / train['total_records']

# 计算测试数据中的 anomaly_ratio
y_test['total_records'] = y_test['binary_label_normal'] + y_test['binary_label_attack']
y_test['anomaly_ratio'] = y_test['binary_label_attack'] / y_test['total_records']

# 绘制训练集和测试集的 anomaly_ratio 分布柱状图
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(train['anomaly_ratio'], bins=30, color='blue', alpha=0.7)
plt.title('Training Set Anomaly Ratio Distribution')
plt.xlabel('Anomaly Ratio')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(y_test['anomaly_ratio'], bins=30, color='green', alpha=0.7)
plt.title('Test Set Anomaly Ratio Distribution')
plt.xlabel('Anomaly Ratio')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig("res.png")