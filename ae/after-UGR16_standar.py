import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder, MinMaxScaler
from visual import visual
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
    
class Autoencoder(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=10):
        super(Autoencoder, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
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
            nn.Linear(70, hidden_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            *block(hidden_dim, 70, normalize=False),
            *block(70, 80),
            *block(80, 90),
            *block(90, 100),
            *block(100, 110),
            *block(110, 120),
            *block(120, 130),
            nn.Linear(130, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 准备数据
np.random.seed(42)

# def load_UGR16():
#     raw_x_train = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)
#     x_train = torch.from_numpy(raw_x_train.values).float()
#     y_train = torch.zeros(len(x_train))


#     raw_x_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
#     x_test = torch.from_numpy(raw_x_test.values).float()
#     y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
#     y_test = torch.from_numpy(y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values)
#     return (x_train, y_train), (x_test, y_test)


def load_UGR16():
    # 先进行标准化
    standard_scaler = StandardScaler()
    
    # 加载并预处理训练数据
    raw_x_train = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)
    x_train_standardized = standard_scaler.fit_transform(raw_x_train.values)  # 仅在训练数据上拟合
    
    # 加载并预处理测试数据
    raw_x_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
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
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(
        columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], 
        axis=1
    )
    y_test = y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1)
    print(y_test.value_counts())
    y_test = torch.from_numpy(y_test.values)
    
    y_train = torch.zeros(len(x_train_normalized))  # 假设训练数据全部为正常数据
    
    return (x_train_normalized, y_train), (x_test_normalized, y_test)


name = "after"
(x_train, y_train), (x_test, y_test) = load_UGR16()


y_train = torch.zeros(len(x_train))
train_mnist = SimpleDataset(x_train, y_train)
dataloader = DataLoader(train_mnist, batch_size=10,
                                shuffle=True)

feature_count = x_train.shape[1]

# 模型训练
model = Autoencoder(input_dim=feature_count, hidden_dim=int(feature_count/2))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    for data in dataloader:
        inputs, _ = data
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


criterion = nn.MSELoss()

test_mnist = SimpleDataset(x_test, y_test)
test_dataloader = DataLoader(test_mnist, batch_size=1,
                                shuffle=False)
model.eval()
RMSEs = np.zeros(x_test.shape[0])
for idx, data in enumerate(test_dataloader):
    inputs, _ = data
    outputs = model(inputs)
    loss = criterion(outputs, inputs)
    RMSEs[idx] = loss.item()
np.savetxt(f"{name}.txt", RMSEs)
visual(name, y_test, RMSEs)