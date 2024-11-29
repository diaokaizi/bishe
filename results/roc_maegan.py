import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 定义函数绘制ROC曲线
def plot_roc_curve(plt, dataset, file_paths, name):
    
    for file_path in file_paths:
        print(file_path)
        # 读取数据
        data = pd.read_csv(os.path.join(dataset, file_path, "labels_anomaly_scores.csv"))
        
        # 提取标签和异常分数
        labels = data['label']
        scores = data['anomaly_score']
        
        # 计算ROC曲线和AUC值
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # 使用文件名作为模型名称
        model_name = file_path  # 提取文件名（无扩展名）
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
file_paths = [
    'AutoEncoder',
    'f-anogan',
    'DeepSVDD',
    'SLAD',
    'MAEGAN',
]

fig_ugr6 = plt.figure(figsize=(6, 6))
# 调用函数绘制ROC曲线
plot_roc_curve(fig_ugr6.gca(), "load_UGR16_faac", file_paths, "UGR16")
plt.tight_layout()
plt.savefig("roc_maegan_ugr16.png")
fig_cic2018 = plt.figure(figsize=(6, 6))
plot_roc_curve(fig_cic2018.gca(), "load_cic2018_faac", file_paths, "CIC2018")
plt.tight_layout()
plt.savefig("roc_maegan_cic2018.png")
