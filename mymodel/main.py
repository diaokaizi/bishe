import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.model_zoo import tqdm
import pandas as pd

# 定义单个三层自编码器
# 1. 定义数据集类
class SimpleDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        data: Tensor, 数据特征
        labels: Tensor, 数据标签
        transform: 可选的变换操作
        """
        self.transform = transform
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]

        if self.transform:
            data = self.transform(data)

        return data, labels

    def __len__(self):
        return len(self.data)

# 2. 定义单个自编码器
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

# 3. 定义总自编码器
class TotalAutoEncoder(nn.Module):
    def __init__(self, input_dim=60, hidden_dim=30):
        super(TotalAutoEncoder, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(50, 40),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(40, hidden_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            *block(hidden_dim, 40, normalize=False),
            *block(40, 50),
            nn.Linear(50, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 4. 定义整体模型
class UnsupervisedAnomalyDetectionModel(nn.Module):
    def __init__(self, feature_map):
        """
        feature_map: List[List[int]], 每个子列表包含一个特征簇的特征索引
        """
        super(UnsupervisedAnomalyDetectionModel, self).__init__()
        self.feature_map = feature_map
        self.autoencoders = nn.ModuleList()
        self.feature_indices = []
        
        # 为每个特征簇创建一个自编码器
        for cluster in feature_map:
            input_dim = len(cluster)
            ae = AutoEncoder(input_dim)
            self.autoencoders.append(ae)
            self.feature_indices.append(cluster)
        
        # 总自编码器的输入维度为特征簇数量
        self.total_autoencoder = TotalAutoEncoder(input_dim=len(feature_map), hidden_dim=int(len(feature_map) * 0.5))

    def forward(self, x, mean=None, std=None):
        reconstruction_errors = []
        raw_errors = []
        for ae, indices in zip(self.autoencoders, self.feature_indices):
            # 提取对应的特征
            cluster_input = x[:, indices]
            reconstructed = ae(cluster_input)
            # 计算重构误差（均方误差）
            error = torch.mean((cluster_input - reconstructed) ** 2, dim=1, keepdim=True)  # (batch_size, 1)
            raw_errors.append(error)
        
        # 拼接所有未标准化的重构误差
        raw_errors = torch.cat(raw_errors, dim=1)  # (batch_size, num_clusters)
        
        if mean is not None and std is not None:
            # 使用预先计算的均值和标准差进行标准化
            normalized_errors = (raw_errors - mean) / std
        else:
            # 如果未提供标准化参数，默认为不标准化
            normalized_errors = raw_errors
        
        # 总自编码器处理标准化后的重构误差
        total_reconstructed = self.total_autoencoder(normalized_errors)
        
        return raw_errors, normalized_errors, total_reconstructed

# 5. 定义损失函数
def combined_loss(raw_reconstruction_errors, normalized_reconstruction_errors, total_reconstructed, total_criterion, individual_criterion, alpha=1.0, beta=1.0):
    """
    alpha: 权重因子，用于独立自编码器的重构误差
    beta: 权重因子，用于总自编码器的重构误差
    """
    # 计算独立自编码器的重构误差（未标准化），不进行均值归约
    individual_loss = alpha * individual_criterion(raw_reconstruction_errors, torch.zeros_like(raw_reconstruction_errors))
    
    # 计算总自编码器的重构误差（标准化后的误差），不进行均值归约
    total_loss = beta * total_criterion(total_reconstructed, normalized_reconstruction_errors)
    
    # 总损失
    loss = individual_loss + total_loss
    return loss

# 6. 定义分阶段训练函数

# a. 训练个别自编码器
def train_individual_autoencoders(model, dataloader, num_epochs=50, learning_rate=1e-3, device='cpu'):
    model.to(device)
    individual_criterions = [nn.MSELoss() for _ in model.autoencoders]
    optimizers = [optim.Adam(ae.parameters(), lr=learning_rate) for ae in model.autoencoders]
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = [0.0 for _ in model.autoencoders]
        for batch in dataloader:
            data, _ = batch
            data = data.to(device)
            
            # 对每个自编码器进行训练
            for i, (ae, indices, criterion, optimizer) in enumerate(zip(model.autoencoders, model.feature_indices, individual_criterions, optimizers)):
                cluster_input = data[:, indices]
                reconstructed = ae(cluster_input)
                loss = criterion(reconstructed, cluster_input)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses[i] += loss.item() * data.size(0)
        
        epoch_losses = [loss / len(dataloader.dataset) for loss in epoch_losses]
        print(f'Epoch [{epoch+1}/{num_epochs}] Individual AutoEncoders Losses: {epoch_losses}')
    
    return model

# b. 计算重构误差
def compute_reconstruction_errors(model, dataloader, device='cpu'):
    model.eval()
    all_errors = []
    with torch.no_grad():
        for batch in dataloader:
            data, _ = batch
            data = data.to(device)
            raw_reconstruction_errors = []
            for ae, indices in zip(model.autoencoders, model.feature_indices):
                cluster_input = data[:, indices]
                reconstructed = ae(cluster_input)
                error = torch.mean((cluster_input - reconstructed) ** 2, dim=1, keepdim=True)  # (batch_size, 1)
                raw_reconstruction_errors.append(error)
            raw_errors = torch.cat(raw_reconstruction_errors, dim=1)  # (batch_size, num_clusters)
            all_errors.append(raw_errors.cpu())
    all_errors = torch.cat(all_errors, dim=0)  # (dataset_size, num_clusters)
    return all_errors

# c. 训练总自编码器
def train_total_autoencoder(model, errors, num_epochs=50, learning_rate=1e-3, device='cpu'):
    model.to(device)
    total_autoencoder = model.total_autoencoder
    total_autoencoder.train()
    optimizer = optim.Adam(total_autoencoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    dataset = TensorDataset(errors, torch.zeros(errors.size(0)))  # Dummy labels
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch_errors, _ = batch
            batch_errors = batch_errors.to(device)
            optimizer.zero_grad()
            reconstructed = total_autoencoder(batch_errors)
            loss = criterion(reconstructed, batch_errors)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_errors.size(0)
        
        epoch_loss /= len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}] Total AutoEncoder Loss: {epoch_loss:.4f}')
    
    return model

# d. 整合分阶段训练流程
def train_model_separately(model, dataloader, num_epochs_individual=50, num_epochs_total=50, learning_rate=1e-3, device='cpu'):
    # 第一阶段：训练个别自编码器
    print("开始训练个别自编码器...")
    model = train_individual_autoencoders(model, dataloader, num_epochs=num_epochs_individual, learning_rate=learning_rate, device=device)
    
    # 第二阶段：计算重构误差
    print("计算重构误差...")
    errors = compute_reconstruction_errors(model, dataloader, device=device)
    
    # 标准化重构误差（使用训练集的均值和标准差）
    mean = errors.mean(dim=0, keepdim=True)
    std = errors.std(dim=0, keepdim=True) + 1e-8  # 防止除以零
    normalized_errors = (errors - mean) / std
    
    # 将标准化参数保存，以便在检测时使用
    torch.save({'mean': mean, 'std': std}, 'results/normalization_params.pth')
    
    # 训练总自编码器
    print("开始训练总自编码器...")
    model.total_autoencoder = train_total_autoencoder(model, normalized_errors, num_epochs=num_epochs_total, learning_rate=learning_rate, device=device)
    
    return model

# 7. 定义异常检测函数
def detect_anomalies_separately(model, dataloader, device='cpu', output_path="results/score.csv"):
    model.eval()
    individual_criterion = nn.MSELoss(reduction='mean')
    total_criterion = nn.MSELoss(reduction='mean')
    
    # 加载标准化参数
    norm_params = torch.load('results/normalization_params.pth', map_location=device)
    mean = norm_params['mean'].to(device)
    std = norm_params['std'].to(device)
    
    with torch.no_grad():
        with open(output_path, "a") as f:
            for (data, labels) in tqdm(dataloader, desc="检测异常"):
                data = data.to(device)
                raw_reconstruction_errors, normalized_errors, total_reconstructed = model(data, mean, std)
                
                # 计算损失
                individual_loss = individual_criterion(raw_reconstruction_errors, torch.zeros_like(raw_reconstruction_errors).to(device))
                total_loss = total_criterion(total_reconstructed, normalized_errors)
                loss = individual_loss + total_loss  # (batch_size,)
                
                # 记录结果
                for l, lbl in zip(loss, labels):
                    f.write(f"{lbl.item()},{l.item()}\n")
    return

def load_UGR16_StandardScaler():
    # 先进行标准化
    standard_scaler = StandardScaler()
    
    # 加载并预处理训练数据
    raw_x_train = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)
    x_train_standardized = standard_scaler.fit_transform(raw_x_train.values)  # 仅在训练数据上拟合
    
    # 加载并预处理测试数据
    raw_x_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
    x_test_standardized = standard_scaler.transform(raw_x_test.values)  # 使用相同的缩放器进行转换
    
    x_train_normalized = torch.from_numpy(x_train_standardized).float()
    
    x_test_normalized = torch.from_numpy(x_test_standardized).float()
    
    # 加载并处理测试标签
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(
        columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], 
        axis=1
    )
    y_test = y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1)
    print(y_test.value_counts())
    y_test = torch.from_numpy(y_test.values)
    
    y_train = torch.zeros(len(x_train_normalized))  # 假设训练数据全部为正常数据
    
    return (x_train_normalized, y_train), (x_test_normalized, y_test)

# 主程序
if __name__ == "__main__":
    # 生成示例数据
    feature_map = [[32, 68, 16, 113, 84], [32, 34, 68, 16, 86], [64, 38, 10, 77, 25], [2, 5, 70, 111, 18], [6, 79, 114, 58, 27], [33, 70, 111, 85, 123], [1, 67, 4, 107, 15], [101, 81, 49, 88, 29], [4, 42, 108, 56, 94], [0, 3, 70, 111, 18], [71, 72, 73, 60, 61], [99, 47, 117, 120, 125], [37, 70, 111, 89, 123], [133, 103, 51, 126, 127], [66, 69, 14, 17, 118], [96, 133, 44, 126, 127], [133, 41, 63, 93, 127], [8, 9, 10, 11, 12], [98, 132, 133, 46, 126], [65, 133, 13, 126, 127], [38, 9, 10, 77, 25], [102, 105, 50, 53, 29], [122, 103, 78, 22, 26], [38, 8, 9, 10, 90], [128, 100, 133, 48, 127], [97, 1, 4, 45, 116], [122, 103, 74, 22, 26], [103, 74, 110, 22, 122], [36, 6, 112, 49, 81, 114, 88, 58, 29], [32, 7, 84, 122, 59], [133, 43, 95, 126, 127], [32, 68, 16, 84, 59], [1, 76, 109, 24, 57], [133, 39, 91, 126, 127], [133, 41, 93, 126, 127], [133, 103, 120, 126, 127], [133, 109, 80, 28, 127], [70, 111, 18, 116, 123], [130, 69, 17, 118, 119], [70, 107, 111, 18, 116], [131, 116, 117, 120, 125], [128, 133, 120, 126, 127], [129, 1, 4, 121, 124], [130, 117, 119, 120, 125], [109, 83, 117, 120, 31], [132, 133, 117, 120, 126], [133, 117, 120, 126, 127], [74, 110, 117, 22, 120], [1, 4, 108, 116, 56], [121, 1, 4, 107, 109, 116, 55, 57], [110, 117, 119, 120, 125], [32, 68, 104, 16, 52, 84], [106, 83, 54, 120, 31], [38, 8, 9, 10, 62], [38, 10, 75, 21, 23], [38, 9, 10, 82, 30], [71, 72, 73, 61, 62], [35, 38, 8, 9, 10, 23, 21, 87, 25, 30], [71, 72, 19, 20, 115, 61], [103, 40, 22, 122, 92]]
    (x_train, y_train), (x_test, y_test) = load_UGR16_StandardScaler()
    print(x_train)
    print(x_test)

    # 创建数据集和数据加载器
    dataset = SimpleDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # 初始化模型
    # 初始化模型
    model = UnsupervisedAnomalyDetectionModel(feature_map)
    
    # 选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 训练模型
    model = train_model_separately(
        model,
        dataloader,
        num_epochs_individual=50,  # 个别自编码器训练轮数
        num_epochs_total=50,       # 总自编码器训练轮数
        learning_rate=1e-3,
        device=device
    )
    
    # 保存训练好的模型
    torch.save(model.state_dict(), "results/unsupervised_anomaly_detection_model.pth")
    
    print("ok")
    # # 异常检测
    # # 设置阈值（需根据实际情况调整）
    # threshold = 0.1  # 示例值
    
    # # 使用相同的数据集进行检测（实际应用中应使用新的数据集）
    test_mnist = SimpleDataset(x_test, y_test)
    test_dataloader = DataLoader(test_mnist, batch_size=1,shuffle=False)
    detect_anomalies_separately(
        model,
        test_dataloader,
        device=device,
        output_path="results/score.csv"
    )
    
    # print(f'检测到的异常样本数量: {sum(anomalies)}')