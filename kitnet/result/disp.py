import matplotlib.pyplot as plt
import pandas as pd



y_test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_train_1s.csv")
y_test['total_records'] = y_test['binary_label_normal'] + y_test['binary_label_attack']

# 计算每个样本的异常比例
y_test['anomaly_ratio'] = y_test['binary_label_attack'] / y_test['total_records']

# # 查看结果
print(y_test[['binary_label_normal', 'binary_label_attack', 'total_records', 'anomaly_ratio']])
print(y_test)
print(y_test['anomaly_ratio'].value_counts())
plt.figure(figsize=(10, 6))
plt.hist(y_test['anomaly_ratio'], bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of Anomaly Ratio')
plt.xlabel('Anomaly Ratio')
plt.ylabel('Frequency')
plt.savefig("Ratiotrain.png")


