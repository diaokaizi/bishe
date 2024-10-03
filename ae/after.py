import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder
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


def load_UNSW_Flow():
    train = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_train_use.csv")
    train = train[train['binary_label_attack'] == 0].drop(columns=['timestamp', 'label_background','label_exploits','label_fuzzers','label_reconnaissance','label_dos','label_analysis','label_backdoor','label_shellcode','label_worms','label_other','binary_label_normal','binary_label_attack'], axis=1).values
    x_train = torch.from_numpy(train).float()
    y_train = torch.zeros(len(x_train))

    test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_use.csv")
    y_test = torch.from_numpy(test['binary_label_attack'].apply(lambda x: 0 if x == 0 else 1).values)
    x_test = test.drop(columns=['timestamp', 'label_background','label_exploits','label_fuzzers','label_reconnaissance','label_dos','label_analysis','label_backdoor','label_shellcode','label_worms','label_other','binary_label_normal','binary_label_attack'], axis=1).values #an m-by-n dataset with m observations
    x_test = torch.from_numpy(x_test).float()
    print(x_train)
    print(x_train.shape)
    print(y_train)
    print(x_test)
    print(x_test.shape)
    print(y_test)
    return (x_train, y_train), (x_test, y_test)

def load_UGR16():
    raw_x_train = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)
    x_train = torch.from_numpy(raw_x_train.values).float()
    y_train = torch.zeros(len(x_train))


    raw_x_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
    x_test = torch.from_numpy(raw_x_test.values).float()
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    y_test = torch.from_numpy(y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values)
    return (x_train, y_train), (x_test, y_test)



name = "after"
(x_train, y_train), (x_test, y_test) = load_UGR16()



mean = x_train.mean(axis=0)  # Mean of each feature
std = x_train.std(axis=0)
normalize = NormalizeTransform(mean, std)
train_mnist = SimpleDataset(x_train, y_train,transform=normalize)
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

mean = x_test.mean(axis=0)  # Mean of each feature
std = x_test.std(axis=0)
normalize = NormalizeTransform(mean, std)
test_mnist = SimpleDataset(x_test, y_test,transform=normalize)
test_dataloader = DataLoader(test_mnist, batch_size=1,
                                shuffle=False)
model.eval()
RMSEs = np.zeros(x_test.shape[0])
for idx, data in enumerate(test_dataloader):
    inputs, _ = data
    outputs = model(inputs)
    loss = criterion(outputs, inputs)

    # 打印每个样本的损失
    print(f"Sample {idx}: Loss = {loss.item()}")

    # 记录 RMSE（可选）
    RMSEs[idx] = loss.item()

# 检测异常
np.savetxt(f"{name}.txt", RMSEs)
visual(name, y_test, RMSEs)