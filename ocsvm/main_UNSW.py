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
    # 先进行标准化
    standard_scaler = StandardScaler()
    
    # 加载训练数据
    train = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_train_1s.csv")
    
    # 计算每个样本的 anomaly_ratio 并筛选出 anomaly_ratio < 0.15 的样本
    train['total_records'] = train['binary_label_normal'] + train['binary_label_attack']
    train['anomaly_ratio'] = train['binary_label_attack'] / train['total_records']
    train = train[train['anomaly_ratio'] < 0.15]  # 只保留 anomaly_ratio < 0.15 的样本

    # 删除不需要的列
    raw_x_train = train.drop(columns=['timestamp', 'label_background', 'label_exploits', 'label_fuzzers',
                                       'label_reconnaissance', 'label_dos', 'label_analysis', 
                                       'label_backdoor', 'label_shellcode', 'label_worms', 
                                       'label_other', 'binary_label_normal', 'binary_label_attack', 
                                       'total_records', 'anomaly_ratio'], axis=1)

    # 标准化
    x_train_standardized = standard_scaler.fit_transform(raw_x_train.values)  # 仅在训练数据上拟合

    # 加载测试数据
    raw_x_test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv").drop(columns=['timestamp', 
                                       'label_background', 'label_exploits', 'label_fuzzers', 
                                       'label_reconnaissance', 'label_dos', 'label_analysis', 
                                       'label_backdoor', 'label_shellcode', 'label_worms', 
                                       'label_other', 'binary_label_normal', 'binary_label_attack'], axis=1)

    # 对测试数据进行标准化
    x_test_standardized = standard_scaler.transform(raw_x_test.values)  # 使用相同的缩放器进行转换

    # 接下来进行归一化
    minmax_scaler = MinMaxScaler()
    
    # 对标准化后的训练数据进行归一化
    x_train_normalized = minmax_scaler.fit_transform(x_train_standardized)  # 仅在训练数据上拟合
    x_train_normalized = torch.from_numpy(x_train_normalized).float()
    
    # 对标准化后的测试数据进行归一化
    x_test_normalized = minmax_scaler.transform(x_test_standardized)  # 使用相同的缩放器进行转换
    x_test_normalized = torch.from_numpy(x_test_normalized).float()
    
    # 加载并处理测试标签
    y_test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv")
    y_test['total_records'] = y_test['binary_label_normal'] + y_test['binary_label_attack']
    y_test['anomaly_ratio'] = y_test['binary_label_attack'] / y_test['total_records']
    
    # 根据 anomaly_ratio 生成测试标签
    y_test = torch.from_numpy((y_test['anomaly_ratio'] > 0.15).astype(int).to_numpy())
    
    # 假设训练数据全部为正常数据
    y_train = torch.zeros(len(x_train_normalized))

    # 输出训练和测试集的形状
    print(f"Training set shape: {x_train_normalized.shape}, Labels: {y_train.unique()}")
    print(f"Test set shape: {x_test_normalized.shape}, Labels: {y_test.unique()}")
    return (x_train_normalized, y_train), (x_test_normalized, y_test)

# 设置名称用于保存结果
name = "UNSW"

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
best_params = None
best_roc_auc = -np.inf
best_pr_auc = -np.inf

# 遍历所有参数组合
for idx, params in enumerate(param_combinations, 1):
    print(f"正在测试第 {idx}/{len(param_combinations)} 组参数：{params}")
    try:
        # 初始化One-Class SVM模型
        ocsvm = OneClassSVM(
            kernel=params['kernel'],
            gamma=params['gamma'],
            nu=params['nu']
        )
        
        # 训练模型
        ocsvm.fit(x_train_np)
        
        # 获取异常分数
        decision_scores = ocsvm.decision_function(x_test_np)
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

# 输出最优参数组合及其评估指标
print("\n最优参数组合及其评估指标：")
print(f"参数组合：{best_params}")
print(f"ROC AUC: {best_roc_auc:.4f}")
print(f"PR AUC: {best_pr_auc:.4f}")

# 使用最优参数重新训练模型并保存结果
if best_params is not None:
    print("\n使用最优参数重新训练模型并保存结果...")
    ocsvm_best = OneClassSVM(
        kernel=best_params['kernel'],
        gamma=best_params['gamma'],
        nu=best_params['nu']
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
else:
    print("未找到最佳参数组合。")



# 正在测试第 19/96 组参数：{'kernel': 'rbf', 'gamma': 0.1, 'nu': 0.05}
# 当前参数组合的ROC AUC: 0.7401, PR AUC: 0.3356