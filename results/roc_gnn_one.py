import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 定义函数绘制ROC曲线
def plot_roc_curve(plt, file_paths, name):
    for (model_name, file_path) in file_paths:
        print(file_path)
        # 读取数据
        data = pd.read_csv(file_path)
        
        # 提取标签和异常分数
        labels = data['label']
        scores = data['anomaly_score']
        
        # 计算ROC曲线和AUC值
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # 使用文件名作为模型名称
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # 图形美化
    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 对角线
    plt.set_xlim([0.0, 1.0])
    plt.set_ylim([0.0, 1.05])
    plt.set_xlabel('False Positive Rate', fontsize=12)
    plt.set_ylabel('True Positive Rate', fontsize=12)
    plt.set_title(f'ROC Curve for {name}', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid()

# 定义文件路径列表（将以下文件路径替换为实际路径）
file_paths_ugr6 = [
    ('FAAC', '/root/bishe/results/load_UGR16_faac/MAEGAN/labels_anomaly_scores.csv'),
    ('GCN', '/root/bishe/results/load_UGR16_GCN/MAEGAN/labels_anomaly_scores.csv'),
    ('GAT', '/root/bishe/results/load_UGR16_GAT/MAEGAN/labels_anomaly_scores.csv'),
    ('GCN-LSTM', '/root/bishe/results/load_UGR16_GCN_LSTM/MAEGAN/labels_anomaly_scores.csv'),
    ('Evolve-GCN', '/root/bishe/results/load_UGR16_Evolve_GCN/MAEGAN/labels_anomaly_scores.csv'),
    ('FG-DyGAT', '/root/bishe/results/load_UGR16_DyGAT/MAEGAN/labels_anomaly_scores.csv'),
]
file_paths_cic2018 = [
    ('FAAC', '/root/bishe/results/load_cic2018_faac/MAEGAN/labels_anomaly_scores.csv'),
    ('GCN', '/root/bishe/results/load_CIC2018_GCN/MAEGAN/labels_anomaly_scores.csv'),
    ('GAT', '/root/bishe/results/load_CIC2018_GAT/MAEGAN/labels_anomaly_scores.csv'),
    ('GCN-LSTM', '/root/bishe/results/load_CIC2018_GCN_LSTM/MAEGAN/labels_anomaly_scores.csv'),
    ('Evolve-GCN', '/root/bishe/results/load_CIC2018_Evolve_GCN/MAEGAN/labels_anomaly_scores.csv'),
    ('FG-DyGAT', '/root/bishe/results/load_cic2018_DyGAT/MAEGAN/labels_anomaly_scores.csv'),
]

# 单独绘制UGR16的ROC曲线
fig_ugr6 = plt.figure(figsize=(6, 6))
plot_roc_curve(fig_ugr6.gca(), file_paths_ugr6, "UGR16")
plt.tight_layout()
plt.savefig("roc_gnn_ugr16.png")

# 单独绘制CIC2018的ROC曲线
fig_cic2018 = plt.figure(figsize=(6, 6))
plot_roc_curve(fig_cic2018.gca(), file_paths_cic2018, "CIC2018")
plt.tight_layout()
plt.savefig("roc_gnn_cic2018.png")
