import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import MinMaxScaler

def load_UGR16_faac():
    df = pd.read_csv("/root/bishe/data/ugr16.csv")
    
    # 定义标签
    label_columns = ['label_dos', 'label_scan11', 'label_scan44', 'label_nerisbotnet']
    label = (df[label_columns].sum(axis=1) > 0).astype(int)
    
    # 去除不需要的特征列
    features_to_drop = [
        'timestamp', 'label_background', 'label_dos', 'label_scan44', 
        'label_scan11', 'label_nerisbotnet', 'label_blacklist', 
        'label_anomaly-udpscan', 'label_anomaly-sshscan', 
        'label_anomaly-spam', 'label_other'
    ]
    df = df.drop(columns=features_to_drop)
    
    # 转换为 NumPy 数组
    x = df.to_numpy()
    y = label.to_numpy()
    
    # 使用 MinMaxScaler 进行归一化
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    
    # 计算皮尔逊相关系数
    pearson_scores = df.corrwith(label).abs()
    
    # 计算卡方检验得分
    chi2_selector = SelectKBest(score_func=chi2, k='all')
    chi2_selector.fit(x, y)
    chi2_scores = pd.Series(chi2_selector.scores_, index=df.columns)
    
    # 合并结果
    feature_importance = pd.DataFrame({
        'Feature': df.columns,
        'Pearson Correlation': pearson_scores,
        'Chi-Squared Score': chi2_scores
    }).sort_values(by=['Pearson Correlation', 'Chi-Squared Score'], ascending=True)
    
    print("Feature Importance Scores:")
    for index, row in feature_importance.iterrows():
        print(f"Feature: {row['Feature']}")
        print(f"  Pearson Correlation: {row['Pearson Correlation']}")
        print(f"  Chi-Squared Score: {row['Chi-Squared Score']}")
        print("-" * 40)

    lowest_60_features = feature_importance.head(75)
    lowest_60_features_list = lowest_60_features['Feature'].tolist()
    print(lowest_60_features_list)
    # 分割数据
    train_len = 1000
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]
    
    return (x_train, y_train), (x_test, y_test), feature_importance


def load_cic2017_faac():
    df = pd.read_csv("/root/bishe/data/cic2017.csv")
    # 定义标签
    label_columns = ['label_DoS_Hulk', 'label_DDoS', 'label_PortScan', 'label_other']
    label = (df[label_columns].sum(axis=1) > 0).astype(int)
    
    # 去除不需要的特征列
    features_to_drop = [
        'timestamp', 'label_background', 'label_DoS_Hulk', 'label_DDoS', 'label_PortScan', 'label_other'
    ]
    df = df.drop(columns=features_to_drop)
    
    # 转换为 NumPy 数组
    x = df.to_numpy()
    y = label.to_numpy()
    
    # 使用 MinMaxScaler 进行归一化
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    
    # 计算皮尔逊相关系数
    pearson_scores = df.corrwith(label).abs()
    
    # 计算卡方检验得分
    chi2_selector = SelectKBest(score_func=chi2, k='all')
    chi2_selector.fit(x, y)
    chi2_scores = pd.Series(chi2_selector.scores_, index=df.columns)
    
    # 合并结果
    feature_importance = pd.DataFrame({
        'Feature': df.columns,
        'Pearson Correlation': pearson_scores,
        'Chi-Squared Score': chi2_scores
    }).sort_values(by=['Pearson Correlation', 'Chi-Squared Score'], ascending=True)
    
    print("Feature Importance Scores:")
    lowest_60_features = feature_importance.head(75)
    lowest_60_features_list = lowest_60_features['Feature'].tolist()

    for index, row in feature_importance.iterrows():
        print(f"Feature: {row['Feature']}")
        print(f"  Pearson Correlation: {row['Pearson Correlation']}")
        print(f"  Chi-Squared Score: {row['Chi-Squared Score']}")
        print("-" * 40)
        if np.isnan(row['Chi-Squared Score']):
            lowest_60_features_list.append(row['Feature'])

    print(lowest_60_features_list)
    print(len(lowest_60_features_list))
    # 分割数据
    train_len = 527
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]
    

    feature_importance = pd.DataFrame({
        'Feature': df.columns,
        'Pearson Correlation': pearson_scores,
        'Chi-Squared Score': chi2_scores
    }).sort_values(by=['Pearson Correlation', 'Chi-Squared Score'], ascending=False)
    
    print("Feature Importance Scores:")
    lowest_60_features = feature_importance.head(20)
    lowest_60_features_list = lowest_60_features['Feature'].tolist()
    print(lowest_60_features_list)

    return (x_train, y_train), (x_test, y_test), feature_importance


def load_cic2018_faac():
    df = pd.read_csv("/root/bishe/data/cic2017.csv")
    # 定义标签
    label_columns = ['label_DoS_Hulk', 'label_DDoS', 'label_PortScan', 'label_other']
    label = (df[label_columns].sum(axis=1) > 0).astype(int)
    
    # 去除不需要的特征列
    features_to_drop = [
        'timestamp', 'label_background', 'label_DoS_Hulk', 'label_DDoS', 'label_PortScan', 'label_other'
    ]
    df = df.drop(columns=features_to_drop)
    
    # 转换为 NumPy 数组
    x = df.to_numpy()
    y = label.to_numpy()
    
    # 使用 MinMaxScaler 进行归一化
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    
    # 计算皮尔逊相关系数
    pearson_scores = df.corrwith(label).abs()
    
    # 计算卡方检验得分
    chi2_selector = SelectKBest(score_func=chi2, k='all')
    chi2_selector.fit(x, y)
    chi2_scores = pd.Series(chi2_selector.scores_, index=df.columns)
    

    feature_importance = pd.DataFrame({
        'Feature': df.columns,
        'Pearson Correlation': pearson_scores,
        'Chi-Squared Score': chi2_scores
    }).sort_values(by=['Pearson Correlation', 'Chi-Squared Score'], ascending=False)
    
    print("Feature Importance Scores:")
    lowest_60_features = feature_importance.head(20)
    lowest_60_features_list = lowest_60_features['Feature'].tolist()
    for index, row in feature_importance.iterrows():
        print(f"Feature: {row['Feature']}")
        print(f"  Pearson Correlation: {row['Pearson Correlation']}")
        print(f"  Chi-Squared Score: {row['Chi-Squared Score']}")
        print("-" * 40)
        if np.isnan(row['Chi-Squared Score']):
            lowest_60_features_list.append(row['Feature'])
    print(lowest_60_features_list)
# 加载数据并计算特征重要性得分
# (x_train, y_train), (x_test, y_test), feature_importance = load_UGR16_faac()
# (x_train, y_train), (x_test, y_test), feature_importance = load_cic2017_faac()
load_cic2018_faac()