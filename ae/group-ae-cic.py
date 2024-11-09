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


def load_UNSW_Flow():
    train = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_train_use.csv")
    train = train[train['binary_label_attack'] == 0].drop(columns=['timestamp', 'label_background','label_exploits','label_fuzzers','label_reconnaissance','label_dos','label_analysis','label_backdoor','label_shellcode','label_worms','label_other','binary_label_normal','binary_label_attack'], axis=1).values
    x_train = torch.from_numpy(train).float()
    y_train = torch.zeros(len(x_train))


def load_UGR16():
    raw_x_train = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)
    x_train = torch.from_numpy(raw_x_train.values).float()
    y_train = torch.zeros(len(x_train))


    raw_x_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
    x_test = torch.from_numpy(raw_x_test.values).float()
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    y_test = torch.from_numpy(y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values)
    return (x_train, y_train), (x_test, y_test)

def load_group():
    seq_len=5
    embs_path = "/root/GCN/DyGCN/data/data/cic2017/model-DGC5-2.pt"
    labels_path = "/root/GCN/DyGCN/data/data/cic2017/labels.npy"
    train_len=[0, 527]

    data_embs = torch.load(embs_path).detach().cpu().numpy()
    print(len(data_embs))
    print(data_embs.shape)
    print(data_embs[train_len[0]+seq_len:train_len[1]].shape)
    print(np.concatenate([data_embs[:train_len[0]],data_embs[train_len[1]:]]).shape)
    labels = np.load(labels_path, allow_pickle=True)
    labels=labels[seq_len:]
    labels=np.concatenate((labels[:train_len[0]], labels[train_len[1]:])).astype(int)
    print(labels)
    x_train = torch.from_numpy(data_embs[train_len[0]+seq_len:train_len[1]])
    test_embs=np.concatenate([data_embs[:train_len[0]],data_embs[train_len[1]:]])
    x_test = torch.from_numpy(test_embs)
    y_test = torch.from_numpy(labels)
    y_train = torch.zeros(len(x_train))
    (x_train, y_train), (x_test, y_test) = (x_train.numpy(), y_train.numpy()), (x_test.numpy(), y_test.numpy())
    minmax_scaler = MinMaxScaler()
    x_train = minmax_scaler.fit_transform(x_train)  # 仅在训练数据上拟合
    x_test = minmax_scaler.transform(x_test)  # 使用相同的缩放器进行转换
    return (x_train, y_train), (x_test, y_test)


name = "CIC"
(x_train, y_train), (x_test, y_test) = load_group()


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

