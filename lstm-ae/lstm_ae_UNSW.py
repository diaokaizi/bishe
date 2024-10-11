import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder
from visual import visual
from torch.utils.model_zoo import tqdm
import matplotlib.pyplot as plt
def load_UNSW():
    
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
    x_train_standardized = torch.from_numpy(raw_x_train.values).float()  # 仅在训练数据上拟合
    
    # 加载测试数据
    raw_x_test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv").drop(columns=['timestamp', 
                                       'label_background', 'label_exploits', 'label_fuzzers', 
                                       'label_reconnaissance', 'label_dos', 'label_analysis', 
                                       'label_backdoor', 'label_shellcode', 'label_worms', 
                                       'label_other', 'binary_label_normal', 'binary_label_attack'], axis=1)

    # 对测试数据进行标准化
    x_test_standardized = torch.from_numpy(raw_x_test.values).float()  # 使用相同的缩放器进行转换

    
    # 加载并处理测试标签
    y_test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv")
    y_test['total_records'] = y_test['binary_label_normal'] + y_test['binary_label_attack']
    y_test['anomaly_ratio'] = y_test['binary_label_attack'] / y_test['total_records']
    
    # 根据 anomaly_ratio 生成测试标签
    y_test = torch.from_numpy((y_test['anomaly_ratio'] > 0.15).astype(int).to_numpy())
    
    # 假设训练数据全部为正常数据
    y_train = torch.zeros(len(x_train_standardized))

    # 输出训练和测试集的形状
    print(f"Training set shape: {x_train_standardized.shape}, Labels: {y_train.unique()}")
    print(f"Test set shape: {x_train_standardized.shape}, Labels: {y_test.unique()}")
    return (x_train_standardized, y_train), (x_test_standardized, y_test)



class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


class LSTMAutoencoder(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 编码器
        self.encoder = nn.LSTM(input_size=num_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # 解码器
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, num_features)
    
    def forward(self, x):
        # 编码
        _, (hidden, cell) = self.encoder(x)
        
        # Repeat the hidden state for each time step in the window
        hidden_repeated = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)  # (batch, seq_len, hidden_dim)
        
        # 解码
        decoded, _ = self.decoder(hidden_repeated)
        
        # 输出
        out = self.output_layer(decoded)
        return out

def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size + 1):
        seq = data[i:i+window_size]
        sequences.append(seq)
    return np.array(sequences)

name = "UNSW"
(x_train, y_train), (x_test, y_test) = load_UNSW()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
window_size = 3  # 根据数据特性选择合适的窗口大小
x_train_sequences = create_sequences(x_train, window_size)
x_test_sequences = create_sequences(x_test, window_size)
train_dataset = TimeSeriesDataset(x_train_sequences)
val_dataset = TimeSeriesDataset(x_test_sequences)
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print("训练集样本数:", len(train_dataset))
print("验证集样本数:", len(val_dataset))

# 定义模型参数
num_features = x_train_sequences.shape[2]
hidden_dim = 64
num_layers = 1

model = LSTMAutoencoder(num_features=num_features, hidden_dim=hidden_dim, num_layers=num_layers)
print(model)

# 4. 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用的设备:", device)

model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.size(0)
    return train_loss / len(dataloader.dataset)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch)
            val_loss += loss.item() * batch.size(0)
    return val_loss / len(dataloader.dataset)

num_epochs = 200
train_losses = []
val_losses = []

for epoch in range(1, num_epochs + 1):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate_epoch(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# 可视化训练过程
plt.figure(figsize=(10,6))
plt.plot(range(1, num_epochs +1), train_losses, label='训练损失')
plt.plot(range(1, num_epochs +1), val_losses, label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('损失 (MSE)')
plt.title('训练过程')
plt.legend()
plt.show()

# 5. 计算重构误差并检测异常
def get_reconstruction_errors(model, dataloader, criterion, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            loss = torch.mean((output - batch) ** 2, dim=(1,2))
            errors.extend(loss.cpu().numpy())
    return np.array(errors)

# 计算所有序列的重构误差
all_errors = get_reconstruction_errors(model, val_loader, criterion, device)

print(all_errors.shape)
np.savetxt(f"{name}.txt", all_errors)
# # 设置阈值（例如，95%的分位数）
# threshold = np.percentile(all_errors, 95)
# print("重构误差阈值:", threshold)

# # 标记异常
# anomalies = all_errors > threshold
# print("检测到的异常数量:", np.sum(anomalies))

# # 标记异常时间步
# anomaly_flags = np.zeros(len(scaled_data))
# for i, is_anomaly in enumerate(anomalies):
#     if is_anomaly:
#         anomaly_flags[i:i+window_size] = 1

# print("检测到的异常时间步数量:", np.sum(anomaly_flags))

# # 6. 可视化结果

# # 可视化重构误差分布
# plt.figure(figsize=(10,6))
# plt.hist(all_errors, bins=50, alpha=0.75, label='重构误差')
# plt.axvline(threshold, color='r', linestyle='--', label='阈值')
# plt.title('重构误差分布')
# plt.xlabel('均方误差')
# plt.ylabel('频数')
# plt.legend()
# plt.show()

# # 可视化异常检测结果（以第一个特征为例）
# plt.figure(figsize=(15,6))
# plt.plot(scaled_data[:,0], label='特征1')  # 选择一个特征进行展示
# plt.scatter(np.where(anomaly_flags == 1), scaled_data[anomaly_flags == 1,0], color='r', label='异常')
# plt.title('异常检测结果')
# plt.xlabel('时间步')
# plt.ylabel('特征1 (标准化)')
# plt.legend()
# plt.show()