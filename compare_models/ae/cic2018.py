import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder, MinMaxScaler
import sys 
import os
root_dir = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(root_dir)
from sklearn.ensemble import IsolationForest
import report_result
import read_data
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
torch.manual_seed(42)

model_name = "AutoEncoder"
# (x_train, y_train), (x_test, y_test) = read_data.load_UGR16_faac()
# filepath = "load_UGR16_faac"
# (x_train, y_train), (x_test, y_test) = read_data.load_cic2017_faac()
# filepath = "load_cic2017_faac"
(x_train, y_train), (x_test, y_test) = read_data.load_cic2018_faac()
filepath = "load_cic2018_faac"

mean = x_train.mean(axis=0)  # Mean of each feature
std = x_train.std(axis=0)
normalize = NormalizeTransform(mean, std)
y_train = torch.zeros(len(x_train))
train_mnist = SimpleDataset(x_train, y_train, normalize)
dataloader = DataLoader(train_mnist, batch_size=10,
                                shuffle=False)

feature_count = x_train.shape[1]

# 模型训练
model = Autoencoder(input_dim=feature_count, hidden_dim=int(feature_count*0.5))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2)

num_epochs = 10
import time
start_time = time.time()  # 记录结束时间
for epoch in range(num_epochs):
    for data in dataloader:
        inputs, _ = data
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
end_time = time.time()  # 记录结束时间
epoch_duration = (end_time - start_time)  # 计算该 epoch 的时间并转换为毫秒
print(f'Average Training Time per Epoch: {epoch_duration:.2f} ms')


criterion = nn.MSELoss()
# mean = x_test.mean(axis=0)  # Mean of each feature
# std = x_test.std(axis=0)
# normalize = NormalizeTransform(mean, std)
test_mnist = SimpleDataset(x_test, y_test, normalize)
test_dataloader = DataLoader(test_mnist, batch_size=1,
                                shuffle=False)
model.eval()
RMSEs = np.zeros(x_test.shape[0])
for idx, data in enumerate(test_dataloader):
    inputs, _ = data
    outputs = model(inputs)
    loss = criterion(outputs, inputs)
    RMSEs[idx] = loss.item()
end_time = time.time()  # 记录结束时间
epoch_duration = (end_time - start_time)  # 计算该 epoch 的时间并转换为毫秒
print(f'test time: {epoch_duration:.2f} ms')
report_result.report_result(model=model_name, name=filepath, anomaly_score=RMSEs, labels=y_test)
