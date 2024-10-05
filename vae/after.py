import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from visual import visual  # 请确保您有一个名为visual的模块，并且其中包含visual函数

# 设置随机种子以确保结果的可重复性
np.random.seed(42)
torch.manual_seed(42)

class NormalizeTransform:
    """ Normalize features with mean and standard deviation. """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = np.where(std == 0, 1, std)

    def __call__(self, sample):
        return (sample - self.mean) / self.std

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
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

class VAE(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=10, latent_dim=5):
        super(VAE, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 130),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(130, 120),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(120, 110),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(110, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 90),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(90, 80),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(80, 70),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 潜在变量的均值和对数方差
        self.fc_mu = nn.Linear(70, latent_dim)
        self.fc_logvar = nn.Linear(70, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 70),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(70, 80),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(80, 90),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(90, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 110),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(110, 120),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(120, 130),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(130, input_dim),
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # 与std形状相同的标准正态分布噪声
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

# 准备数据
def load_UGR16():
    raw_x_train = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)
    x_train = torch.from_numpy(raw_x_train.values).float()
    y_train = torch.zeros(len(x_train))
    
    raw_x_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
    x_test = torch.from_numpy(raw_x_test.values).float()
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(
        columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], 
        axis=1
    )
    y_test = torch.from_numpy(y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values)
    print("测试集标签分布：")
    print(y_test.bincount())
    return (x_train, y_train), (x_test, y_test)

name = "after"
(x_train, y_train), (x_test, y_test) = load_UGR16()

# 计算训练数据的均值和标准差
mean = x_train.mean(axis=0)  # 每个特征的均值
std = x_train.std(axis=0)    # 每个特征的标准差
normalize = NormalizeTransform(mean, std)
train_dataset = SimpleDataset(x_train, y_train, transform=normalize)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 模型参数
input_dim = x_train.shape[1]
hidden_dim = 10
latent_dim = int(x_train.shape[1]/ 2)

# 初始化模型、损失函数和优化器
model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# VAE的损失函数包括重构损失和KL散度
def loss_function(recon_x, x, mu, logvar):
    # 重构损失（MSE）
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # KL散度
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 250
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(inputs)
        loss = loss_function(recon_batch, inputs, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    average_loss = train_loss / len(train_loader)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')
        # 评估模型
        model.eval()
        test_dataset = SimpleDataset(x_test, y_test, transform=normalize)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        reconstruction_errors = []
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                recon_batch, mu, logvar = model(inputs)
                # 计算重构误差
                loss = nn.functional.mse_loss(recon_batch, inputs, reduction='mean').item()
                reconstruction_errors.append(loss)
                if (idx + 1) % 1000 == 0 or idx == 0:
                    print(f"Processed {idx+1}/{len(test_loader)} samples")

        # 转换为NumPy数组
        reconstruction_errors = np.array(reconstruction_errors)

        # 保存重构误差
        np.savetxt(f"{name}_{epoch}.txt", reconstruction_errors)

        # 可视化结果
        visual(name, y_test, reconstruction_errors)