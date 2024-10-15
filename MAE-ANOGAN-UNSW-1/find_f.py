import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def load_UNSW():
    # 加载训练数据
    train = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_train_1s.csv")
    
    # 计算 anomaly_ratio
    train['total_records'] = train['binary_label_normal'] + train['binary_label_attack']
    train['anomaly_ratio'] = train['binary_label_attack'] / train['total_records']
    
    # 分配标签：anomaly_ratio >= 0.11 为异常 (1)，否则为正常 (0)
    train['label'] = (train['anomaly_ratio'] >= 0.11).astype(int)
    
    # 特征选择：删除不需要的列
    feature_columns = ['feature1', 'feature2', 'feature3', ...]  # 请根据实际数据集填写
    raw_x_train = train.drop(columns=['timestamp', 'label_background', 'label_exploits', 'label_fuzzers',
                                      'label_reconnaissance', 'label_dos', 'label_analysis', 
                                      'label_backdoor', 'label_shellcode', 'label_worms', 
                                      'label_other', 'binary_label_normal', 'binary_label_attack', 
                                      'total_records', 'anomaly_ratio', 'label'], axis=1)
    y_train = train['label'].values
    
    # 加载测试数据
    test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv")
    test['total_records'] = test['binary_label_normal'] + test['binary_label_attack']
    test['anomaly_ratio'] = test['binary_label_attack'] / test['total_records']
    test['label'] = (test['anomaly_ratio'] >= 0.11).astype(int)
    
    raw_x_test = test.drop(columns=['timestamp', 'label_background', 'label_exploits', 'label_fuzzers', 
                                    'label_reconnaissance', 'label_dos', 'label_analysis', 
                                    'label_backdoor', 'label_shellcode', 'label_worms', 
                                    'label_other', 'binary_label_normal', 'binary_label_attack', 
                                    'total_records', 'anomaly_ratio', 'label'], axis=1)
    y_test = test['label'].values
    
    # 特征归一化
    scaler = MinMaxScaler()
    x_train_normalized = scaler.fit_transform(raw_x_train)
    x_test_normalized = scaler.transform(raw_x_test)
    
    print(f"训练集形状: {x_train_normalized.shape}, 标签分布: {np.unique(y_train, return_counts=True)}")
    print(f"测试集形状: {x_test_normalized.shape}, 标签分布: {np.unique(y_test, return_counts=True)}")
    
    return (x_train_normalized, y_train), (x_test_normalized, y_test), raw_x_train.columns

def train_and_evaluate():
    (x_train, y_train), (x_test, y_test), feature_names = load_UNSW()
    
    # 训练随机森林分类器
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(x_train, y_train)
    
    # 在测试集上进行预测
    y_pred = clf.predict(x_test)
    
    # 评估模型
    print("分类报告:")
    print(classification_report(y_test, y_pred))
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
    # 获取特征重要性
    importances = clf.feature_importances_
    feature_importances = pd.Series(importances, index=feature_names)
    feature_importances = feature_importances.sort_values(ascending=False)
    
    # 可视化前20个重要特征
    print(feature_importances[:20])
    for t in list(feature_importances):
        print(t)
    
    return clf, feature_importances

if __name__ == "__main__":
    clf, feature_importances = train_and_evaluate()