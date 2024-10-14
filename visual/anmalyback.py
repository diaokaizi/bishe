import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

def log_transform(scores):
    # 对所有数值进行对数变换，避免对零或负数取对数
    return np.log1p(scores)

def normalize_scores(anomaly_score):
    # 将异常得分标准化到0-1区间
    anomaly_score = (anomaly_score - np.mean(anomaly_score)) / np.std(anomaly_score)  # 标准化
    scaler = MinMaxScaler(feature_range=(0, 1))  # 初始化Min-Max缩放器
    normalized_scores = scaler.fit_transform(anomaly_score.reshape(-1, 1)).flatten()  # 归一化到[0, 1]范围
    return normalized_scores

def augment_anomaly_samples(labels, anomaly_score, repeat_factor=50):
    # 分离正常和异常样本
    normal_scores = anomaly_score[labels == 0]
    anomaly_scores = anomaly_score[labels == 1]

    # 重复异常样本
    augmented_anomaly_scores = np.tile(anomaly_scores, repeat_factor)
    augmented_anomaly_labels = np.tile([1], len(anomaly_scores) * repeat_factor)

    # 将正常样本与扩展的异常样本合并
    augmented_scores = np.concatenate([normal_scores, augmented_anomaly_scores])
    augmented_labels = np.concatenate([labels[labels == 0], augmented_anomaly_labels])

    return augmented_labels, augmented_scores

def plot_augmented_anomaly_score(name, labels, anomaly_score, anomaly_columns, y_test, repeat_factor=30):
    # 进行异常样本的增强
    augmented_labels, augmented_scores = augment_anomaly_samples(labels, anomaly_score, repeat_factor)
    augmented_scores = normalize_scores(augmented_scores)

    # 分离增强后的正常和异常样本
    normal_scores = augmented_scores[augmented_labels == 0]
    anomaly_scores = augmented_scores[augmented_labels == 1]

    # 创建绘图
    plt.figure(figsize=(12, 8))

    # 使用 Seaborn 绘制平滑的 KDE 曲线
    sns.kdeplot(normal_scores, label="Normal", color='blue', linewidth=2)
    sns.kdeplot(anomaly_scores, label="Augmented Anomaly", color='red', linewidth=2)
    print(anomaly_columns)
    # 对每个异常类型分别绘制KDE曲线
    # for anomaly_type in anomaly_columns:
    #     anomaly_type_scores = anomaly_score[y_test[anomaly_type] > 0]  # 选择特定异常类型的样本
    #     sns.kdeplot(anomaly_type_scores, label=anomaly_type, linewidth=2)

    # 设置图形标题和标签
    plt.title(f'Augmented Anomaly Score Distribution with Specific Anomalies for {name}')
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.legend()

def remove_top_n_rows(df, y_test, column_name, n=2):
    # 找到 'anomaly_score' 列中最大的 n 行的索引
    top_n_indices = df.nlargest(n, column_name).index
    # 删除 df 和 y_test 中对应的行
    df = df.drop(top_n_indices)
    y_test = y_test.drop(top_n_indices)
    return df, y_test

def fgan(path, name):
    df = pd.read_csv(path)
    
    # 读取 y_test 数据（包括异常类型标签）
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv")
    
    # 删除 anomaly_score 最大的两百个样本，并同步删除 y_test 中对应的行
    df, y_test = remove_top_n_rows(df, y_test, 'anomaly_score', n=200)
    
    # 提取 labels 和 anomaly_score
    trainig_label = 0
    labels = np.where(df["label"].values == trainig_label, 0, 1)
    anomaly_score = df["anomaly_score"].values
    
    # 归一化异常得分
    log_anomaly_score = log_transform(anomaly_score)
    final_score = normalize_scores(log_anomaly_score)

    # 定义异常类型的列名
    anomaly_columns = ['labeldos', 'labelscan11', 'labelscan44', 'labelnerisbotnet']

    # 绘制增强后的得分分布图，并绘制每类异常得分的曲线
    plot_augmented_anomaly_score(name, labels, final_score, anomaly_columns, y_test)

# 调用 fgan 函数，绘制异常得分分布图
fgan("/root/bishe/MAE-ANOGAN2/results/score.csv", "MAE-ANOGAN")

plt.legend()
plt.savefig("anomaly.png")
