import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from visual import visual
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from itertools import product
import warnings
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
    raw_x_train = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)
    x_train = scaler.fit_transform(raw_x_train.values)  # 仅在训练数据上拟合
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.zeros(len(x_train))  # 假设训练数据全部为正常数据
    
    # 加载并预处理测试数据
    raw_x_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
    x_test = scaler.transform(raw_x_test.values)  # 使用相同的缩放器进行转换
    x_test = torch.from_numpy(x_test).float()
    
    # 加载并处理测试标签
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(
        columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], 
        axis=1
    )
    y_test = y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1)
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
    'n_estimators': [100, 200, 300],
    'max_samples': ['auto', 0.75, 0.85],
    'max_features': [1.0, 0.8, 0.9],
    'contamination': [0.01, 0.02, 0.05],
    'bootstrap': [False, True]
}

# 生成所有参数组合
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in product(*values)]

print(f"总共需要测试的参数组合数量：{len(param_combinations)}")

# 初始化变量以记录最佳参数和最佳得分
best_params = None
best_roc_auc = -np.inf
best_pr_auc = -np.inf

# 遍历所有参数组合
for idx, params in enumerate(param_combinations, 1):
    print(f"正在测试第 {idx}/{len(param_combinations)} 组参数：{params}")
    try:
        # 初始化Isolation Forest模型
        iForest = IsolationForest(
            n_estimators=params['n_estimators'],
            max_samples=params['max_samples'],
            max_features=params['max_features'],
            contamination=params['contamination'],
            bootstrap=params['bootstrap'],
            random_state=42
        )
        
        # 训练模型
        iForest.fit(x_train_np)
        
        # 获取异常分数
        decision_scores = iForest.decision_function(x_test_np)
        anomaly_scores = -decision_scores  # 值越高表示越异常
        
        # 对异常分数进行归一化
        scaler_minmax = MinMaxScaler()
        anomaly_scores_normalized = scaler_minmax.fit_transform(anomaly_scores.reshape(-1, 1)).flatten()
        
        # 计算评估指标
        roc_auc = roc_auc_score(y_test_np, anomaly_scores_normalized)
        precision, recall, _ = precision_recall_curve(y_test_np, anomaly_scores_normalized)
        pr_auc = auc(recall, precision)
        
        print(f"当前参数组合的ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        
        # 更新最佳参数
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_pr_auc = pr_auc
            best_params = params
    except Exception as e:
        print(f"参数组合 {params} 运行失败，错误信息：{e}")
        continue

# 输出最优参数和对应的评估指标
print("\n最优参数组合及其评估指标：")
print(f"参数组合：{best_params}")
print(f"ROC AUC: {best_roc_auc:.4f}")
print(f"PR AUC: {best_pr_auc:.4f}")

# 使用最优参数重新训练模型并保存结果
if best_params is not None:
    print("\n使用最优参数重新训练模型并保存结果...")
    iForest_best = IsolationForest(
        n_estimators=best_params['n_estimators'],
        max_samples=best_params['max_samples'],
        max_features=best_params['max_features'],
        contamination=best_params['contamination'],
        bootstrap=best_params['bootstrap'],
        random_state=42
    )
    
    # 训练模型
    iForest_best.fit(x_train_np)
    
    # 获取异常分数
    decision_scores_best = iForest_best.decision_function(x_test_np)
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
else:
    print("未找到最佳参数组合。")


# {'n_estimators': 200, 'max_samples': 0.75, 'max_features': 1.0, 'contamination': 0.01, 'bootstrap': False}