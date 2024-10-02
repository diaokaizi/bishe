import numpy as np
import skfuzzy as fuzz
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd

def fix_name():
    return ["sportsocks", "sportprivate", "dportirc", "sporttelnet", "sportrapservice", "dporthttp",
            "sportsyslog", "sportreserved", "dportkpasswd", "tcpflagsACK", "npacketsmedium",
            "sportcups", "dporttelnet", "sportldaps", "tcpflagsPSH", "dportoracle"]

def load_UGR16():
    selected_feature_names = fix_name()
    raw_x_train = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)
    x_train = raw_x_train[selected_feature_names]
    x_train = torch.from_numpy(x_train.values).float()
    y_train = torch.zeros(len(x_train))


    raw_x_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
    x_test = raw_x_test[selected_feature_names]
    x_test = torch.from_numpy(x_test.values).float()
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    y_test = torch.from_numpy(y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values)
    return (x_train, y_train), (x_test, y_test)

class FinAutoencoder(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=10):
        super(Autoencoder, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 70),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(70, 65),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(65, 60),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(60, 55),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(55, 50),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(50, 45),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(45, hidden_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            *block(hidden_dim, 45, normalize=False),
            *block(45, 50),
            *block(50, 60),
            *block(60, 65),
            *block(65, 70),
            nn.Linear(70, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # 定义解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(autoencoder, data_loader, num_epochs=10, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for batch_data in data_loader:
            inputs = batch_data[0]
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 可以添加日志输出
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
def compute_reconstruction_errors(autoencoder, data):
    with torch.no_grad():
        outputs = autoencoder(data)
        # 计算每个样本的重构误差（MSE）
        errors = torch.mean((outputs - data) ** 2, dim=1)
    return errors
def normalize_errors(error_features):
    # error_features 的形状为 (n_samples, n_clusters)
    # 计算每个重构误差特征的均值和标准差
    mean = torch.mean(error_features, dim=0)
    std = torch.std(error_features, dim=0)
    # 防止标准差为 0，添加一个很小的数
    std[std == 0] = 1e-8
    # 标准化
    normalized_errors = (error_features - mean) / std
    return normalized_errors, mean, std

def normalize_data(x, mean=None, std=None):
    # 如果没有提供mean和std，则计算
    if mean is None or std is None:
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        std[std == 0] = 1e-8  # 防止标准差为0
    # 标准化
    normalized_x = (x - mean) / std
    return normalized_x, mean, std


x_train = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtrain.csv").drop(columns=['Row'], axis=1).values #an m-by-n dataset with m observations
print(len(x_train))
feature_map =  [[113], [64], [2, 5], [34, 86], [51], [40, 92], [42, 94], [35, 87], [27, 79], [30, 82], [44, 96], [75], [23], [16, 68], [13, 65], [60], [52], [115], [25, 77], [38], [14, 66], [33, 85], [21, 73], [0, 3], [12], [46, 98], [48, 100], [90], [49, 101], [47, 99], [9, 61, 8, 11, 10, 62], [104], [63], [71, 72, 19, 20], [53, 105, 50, 102], [45, 97], [39, 91], [43, 95], [7, 59], [106], [32, 36, 88], [78], [29], [37, 89], [67], [15], [26], [54], [6, 112, 58, 114], [122], [41, 93], [81, 84], [123], [118, 17, 69], [103], [28, 80, 24, 76], [128], [127], [18, 70], [130], [129], [74], [83], [133, 126, 132, 125, 117, 120, 110, 119, 55, 107], [22, 31], [131], [111, 124, 56, 108, 57, 109, 121, 116, 1, 4]]


x_train_tensor = torch.from_numpy(x_train).float()
x_train_tensor, data_mean, data_std = normalize_data(x_train_tensor)

# 步骤 2：构建自编码器
cluster_autoencoders = []
for idx, feature_indices in enumerate(feature_map):
    input_dim = len(feature_indices)
    if input_dim == 0:
        continue
    autoencoder = Autoencoder(input_dim, int((input_dim + 1) / 2))
    cluster_autoencoders.append({
        'autoencoder': autoencoder,
        'feature_indices': feature_indices
    })
# 步骤 3：训练自编码器
# x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
print(len(cluster_autoencoders))

for cluster in cluster_autoencoders:
    feature_indices = cluster['feature_indices']
    print(feature_indices)
    autoencoder = cluster['autoencoder']
    cluster_data = x_train_tensor[:, feature_indices]
    dataset = torch.utils.data.TensorDataset(cluster_data)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    train_autoencoder(autoencoder, data_loader, num_epochs=30)
# 步骤 4：计算重构误差
reconstruction_errors = []
for cluster in cluster_autoencoders:
    feature_indices = cluster['feature_indices']
    autoencoder = cluster['autoencoder']
    cluster_data = x_train_tensor[:, feature_indices]
    errors = compute_reconstruction_errors(autoencoder, cluster_data)
    reconstruction_errors.append(errors.unsqueeze(1))
error_features = torch.cat(reconstruction_errors, dim=1)

###########################################
(a, b), (c, _) = load_UGR16()
error_features = torch.cat([error_features, a], dim=1)
#####


# **添加归一化处理**
error_features_normalized, error_mean, error_std = normalize_data(error_features)

# 步骤 5：训练最终自编码器
final_input_dim = error_features.shape[1]
final_hidden_dim = int((final_input_dim + 1) / 2)

final_autoencoder = FinAutoencoder(final_input_dim, final_hidden_dim)
error_dataset = torch.utils.data.TensorDataset(error_features_normalized)
error_data_loader = DataLoader(error_dataset, batch_size=32, shuffle=True)
train_autoencoder(final_autoencoder, error_data_loader, num_epochs=100)
# 返回模型和必要的信息







x_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv").drop(columns=['Row'], axis=1).values #an m-by-n dataset with m observations

x_test_tensor = torch.from_numpy(x_test).float()
x_test_normalized = (x_test_tensor - data_mean) / data_std
x_test_normalized[x_test_normalized == float('inf')] = 0
x_test_normalized[x_test_normalized == float('-inf')] = 0
x_test_normalized = torch.nan_to_num(x_test_normalized, nan=0.0, posinf=0.0, neginf=0.0)

# 步骤 2：计算每个自编码器的重构误差
reconstruction_errors = []
for i, cluster in enumerate(cluster_autoencoders):
    feature_indices = cluster['feature_indices']
    autoencoder = cluster['autoencoder']
    cluster_data = x_test_normalized[:, feature_indices]
    errors = compute_reconstruction_errors(autoencoder, cluster_data)
    reconstruction_errors.append(errors.unsqueeze(1))
error_features = torch.cat(reconstruction_errors, dim=1)

###########################################
(a, b), (c, _) = load_UGR16()
error_features = torch.cat([error_features, c], dim=1)
#####

# **使用训练时的均值和标准差进行归一化**
normalized_error_features = (error_features - error_mean) / error_std

# 防止标准差为 0
normalized_error_features[torch.isinf(normalized_error_features)] = 0
normalized_error_features[torch.isnan(normalized_error_features)] = 0

anomaly_scores = compute_reconstruction_errors(final_autoencoder, normalized_error_features).numpy()
np.savetxt(f"res.txt", anomaly_scores)


