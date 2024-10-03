import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 1. 数据加载和预处理
def load_UGR16():
    raw_x_train = pd.read_csv("/root/faac-compare/data/after/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)
    # raw_x_train = raw_x_train[selected_feature_names]
    x_train = torch.from_numpy(raw_x_train.values).float()
    y_train = torch.zeros(len(x_train))


    raw_x_test = pd.read_csv("/root/faac-compare/data/after/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
    # raw_x_test = raw_x_test[selected_feature_names]
    x_test = torch.from_numpy(raw_x_test.values).float()
    y_test = pd.read_csv("/root/faac-compare/data/after/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    y_test = torch.from_numpy(y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values)
    return (x_train, y_train), (x_test, y_test)

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

# 2. 定义改进后的 LSTM Autoencoder
class ImprovedLSTMAutoencoder(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_layers=2, bidirectional=True, dropout=0.2):
        super(ImprovedLSTMAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 编码器（双向 LSTM）
        self.encoder = nn.LSTM(input_size=num_features, hidden_size=hidden_dim, num_layers=num_layers, 
                               batch_first=True, bidirectional=bidirectional, dropout=dropout)
        
        # 解码器（双向 LSTM）
        decoder_input_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.decoder = nn.LSTM(input_size=decoder_input_size, hidden_size=hidden_dim, 
                               num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_features)
    
    def forward(self, x):
        # 编码
        _, (hidden, cell) = self.encoder(x)
        
        if self.bidirectional:
            # 合并双向的隐藏状态和细胞状态
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
            cell = cell[-1,:,:]
        
        # 调整隐藏状态和细胞状态以适应解码器
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = cell.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        # 解码
        decoded, _ = self.decoder(x, (hidden, cell))
        
        # 输出
        out = self.output_layer(decoded)
        return out

# 3. 创建序列
def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size + 1):
        seq = data[i:i+window_size]
        sequences.append(seq)
    return np.array(sequences)

# 4. 训练和验证函数
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

# 5. 获取重构误差
def get_reconstruction_errors(model, dataloader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            loss = torch.mean((output - batch) ** 2, dim=(1,2))  # 计算每个序列的重构误差
            errors.extend(loss.cpu().numpy())
    return np.array(errors)

# 6. 主执行流程
if __name__ == "__main__":
    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_UGR16()
    
    # 标准化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)  # 使用相同的 scaler
    
    # 创建序列
    window_size = 3  # 根据数据特性选择合适的窗口大小
    x_train_sequences = create_sequences(x_train, window_size)
    x_test_sequences = create_sequences(x_test, window_size)
    
    # 创建数据集和数据加载器
    train_dataset = TimeSeriesDataset(x_train_sequences)
    val_dataset = TimeSeriesDataset(x_test_sequences)
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print("训练集样本数:", len(train_dataset))
    print("验证集样本数:", len(val_dataset))
    
    # 定义模型参数
    num_features = 134  # 更新为134
    hidden_dim = 64     # 增加隐藏维度
    num_layers = 2      # 增加LSTM层数
    bidirectional = True  # 使用双向LSTM
    dropout = 0.2         # 添加Dropout
    
    # 初始化模型
    model = ImprovedLSTMAutoencoder(num_features=num_features, hidden_dim=hidden_dim, num_layers=num_layers, 
                                   bidirectional=bidirectional, dropout=dropout)
    print(model)
    
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用的设备:", device)
    
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 定义学习率调度器和早停
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    patience = 20
    best_val_loss = float('inf')
    counter = 0
    
    # 训练模型
    num_epochs = 200
    train_losses = []
    val_losses = []
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # 调度器步进
        scheduler.step(val_loss)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_lstm_autoencoder.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break
    
    # 绘制训练和验证损失
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='训练损失')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失 (MSE)')
    plt.title('训练过程')
    plt.legend()
    plt.show()
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_lstm_autoencoder.pth'))
    
    # 计算重构误差
    train_errors = get_reconstruction_errors(model, train_loader, device)
    test_errors = get_reconstruction_errors(model, val_loader, device)
    
    # 选择阈值（例如，训练集的95%分位数）
    threshold = np.percentile(train_errors, 95)
    print(f"重构误差阈值: {threshold}")
    
    # 标记异常
    y_pred = (test_errors > threshold).astype(int)
    print(f"检测到的异常数量: {np.sum(y_pred)}")
    
    # 计算评估指标
    auc_score = roc_auc_score(y_test, test_errors)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"ROC-AUC Score: {auc_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # 绘制 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_test, test_errors)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Anomaly Detection')
    plt.legend(loc='lower right')
    plt.show()
    
    # 保存重构误差和预测结果
    name = "123"
    np.savetxt(f"{name}_reconstruction_errors_train.txt", train_errors)
    np.savetxt(f"{name}_reconstruction_errors_test.txt", test_errors)
    np.savetxt(f"{name}_predictions_test.txt", y_pred)