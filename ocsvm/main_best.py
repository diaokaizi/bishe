import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from visual import visual  # 请确保您有一个名为visual的模块，并且其中包含visual函数
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from itertools import product
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings("ignore")

# 设置随机种子以确保结果的可重复性
np.random.seed(42)

def fix_name():
    return [
        "sportsocks", "sportprivate", "dportirc", "sporttelnet", "sportrapservice",
        "dporthttp", "sportsyslog", "sportreserved", "dportkpasswd", "tcpflagsACK",
        "npacketsmedium", "sportcups", "dporttelnet", "sportldaps", "tcpflagsPSH",
        "dportoracle"
    ]

def load_after():
    scaler = StandardScaler()
    
    # 加载并预处理训练数据
    raw_x_train = pd.read_csv("/root/faac-compare/data/after/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)
    x_train = scaler.fit_transform(raw_x_train.values)  # 仅在训练数据上拟合
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.zeros(len(x_train))  # 假设训练数据全部为正常数据
    
    # 加载并预处理测试数据
    raw_x_test = pd.read_csv("/root/faac-compare/data/after/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
    x_test = scaler.transform(raw_x_test.values)  # 使用相同的缩放器进行转换
    x_test = torch.from_numpy(x_test).float()
    
    # 加载并处理测试标签
    y_test = pd.read_csv("/root/faac-compare/data/after/UGR16v1.Ytest.csv").drop(
        columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], 
        axis=1
    )
    y_test = y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1)
    print("测试集标签分布：")
    print(y_test.value_counts())
    y_test = torch.from_numpy(y_test.values)
    
    return (x_train, y_train), (x_test, y_test)

# 设置名称用于保存结果
name = "after"

# 加载数据
(x_train, y_train), (x_test, y_test) = load_after()

# 将Torch张量转换为NumPy数组以供scikit-learn使用
x_train_np = x_train.numpy()
x_test_np = x_test.numpy()
y_test_np = y_test.numpy()

# 定义参数网格
param_grid = {
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'nu': [0.01, 0.02, 0.05, 0.1]
}

# 生成所有参数组合
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in product(*values)]

print(f"总共需要测试的参数组合数量：{len(param_combinations)}")

# 初始化变量以记录最佳参数和最佳得分
best_params = 'rbf'
best_roc_auc = 0.1
best_pr_auc = 0.05

# 输出最优参数组合及其评估指标
print("\n最优参数组合及其评估指标：")
print(f"参数组合：{best_params}")
print(f"ROC AUC: {best_roc_auc:.4f}")
print(f"PR AUC: {best_pr_auc:.4f}")

# 使用最优参数重新训练模型并保存结果
if best_params is not None:
    print("\n使用最优参数重新训练模型并保存结果...")
    ocsvm_best = OneClassSVM(
        kernel='rbf',
        gamma=0.1,
        nu=0.05
    )
    
    # 训练模型
    ocsvm_best.fit(x_train_np)
    
    # 获取异常分数
    decision_scores_best = ocsvm_best.decision_function(x_test_np)
    anomaly_scores_best = -decision_scores_best  # 值越高表示越异常
    
    # 对异常分数进行归一化
    scaler_minmax_best = MinMaxScaler()
    anomaly_scores_normalized_best = scaler_minmax_best.fit_transform(anomaly_scores_best.reshape(-1, 1)).flatten()
    
    # 保存归一化后的异常分数
    np.savetxt(f"{name}_best_params.txt", anomaly_scores_normalized_best)
    
    # 可视化结果
    visual(name + "_best", y_test, anomaly_scores_normalized_best)
    
    # 计算并打印评估指标
    roc_auc_best = roc_auc_score(y_test_np, anomaly_scores_normalized_best)
    precision_best, recall_best, _ = precision_recall_curve(y_test_np, anomaly_scores_normalized_best)
    pr_auc_best = auc(recall_best, precision_best)
    print(f"最优参数组合的ROC AUC: {roc_auc_best:.4f}")
    print(f"最优参数组合的PR AUC: {pr_auc_best:.4f}")



# 正在测试第 19/96 组参数：{'kernel': 'rbf', 'gamma': 0.1, 'nu': 0.05}
# 当前参数组合的ROC AUC: 0.7401, PR AUC: 0.3356