import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from fanogan_muti.train_wgangp import train_wgangp
from fanogan_muti.train_encoder_izif import train_encoder_izif
from bishe.MAEGAN.model.model import Generator, Discriminator, Encoder
from tools import SimpleDataset, load_UGR16, NormalizeTransform
import bishe.MAEGAN.model.MAE as kit
import numpy as np
import pandas as pd
from fanogan_muti.test_anomaly_detection import test_anomaly_detection

from sklearn.preprocessing import StandardScaler, MinMaxScaler
def load_UNSW(cn = 1000, ratio = 0.12):
    # 先进行标准化
    
    # 加载训练数据
    train = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_train_1s.csv")
    train = train[~((train['binary_label_normal'] + train['binary_label_attack']) < cn)]
    # 计算每个样本的 anomaly_ratio 并筛选出 anomaly_ratio < 0.15 的样本
    train['total_records'] = train['binary_label_normal'] + train['binary_label_attack']
    train['anomaly_ratio'] = train['binary_label_attack'] / train['total_records']
    train = train[train['anomaly_ratio'] < ratio]  # 只保留 anomaly_ratio < 0.15 的样本

    # 删除不需要的列
    raw_x_train = train.drop(columns=['timestamp', 'label_background', 'label_exploits', 'label_fuzzers',
                                       'label_reconnaissance', 'label_dos', 'label_analysis', 
                                       'label_backdoor', 'label_shellcode', 'label_worms', 
                                       'label_other', 'binary_label_normal', 'binary_label_attack', 
                                       'total_records', 'anomaly_ratio'], axis=1)


    # 加载测试数据
    raw_x_test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv")

    raw_x_test = raw_x_test[~((raw_x_test['binary_label_normal'] + raw_x_test['binary_label_attack']) < cn)].drop(columns=['timestamp', 
                                       'label_background', 'label_exploits', 'label_fuzzers', 
                                       'label_reconnaissance', 'label_dos', 'label_analysis', 
                                       'label_backdoor', 'label_shellcode', 'label_worms', 
                                       'label_other', 'binary_label_normal', 'binary_label_attack'], axis=1)

    # 接下来进行归一化
    minmax_scaler = MinMaxScaler()
    
    # 对标准化后的训练数据进行归一化
    x_train_normalized = minmax_scaler.fit_transform(raw_x_train.values)  # 仅在训练数据上拟合
    
    # 对标准化后的测试数据进行归一化
    x_test_normalized = minmax_scaler.transform(raw_x_test.values)  # 使用相同的缩放器进行转换
    
    # 加载并处理测试标签
    y_test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_1s.csv")
    y_test = y_test[~((y_test['binary_label_normal'] + y_test['binary_label_attack']) < cn)]
    y_test['total_records'] = y_test['binary_label_normal'] + y_test['binary_label_attack']
    y_test['anomaly_ratio'] = y_test['binary_label_attack'] / y_test['total_records']
    
    # 根据 anomaly_ratio 生成测试标签
    y_test = (y_test['anomaly_ratio'] >= ratio).astype(int).to_numpy()
    
    # 假设训练数据全部为正常数据
    y_train = np.zeros(len(x_train_normalized))

    # 输出训练和测试集的形状
    print(f"Training set shape: {x_train_normalized.shape}, Labels: {np.unique(y_train)}")
    print(f"Test set shape: {x_test_normalized.shape}, Labels: {np.unique(y_test)}")
    
    return (x_train_normalized, y_train), (x_test_normalized, y_test)


def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (x_train, y_train), (x_test, y_test) = load_UNSW()
    feature_map = [[0, 3, 75, 110, 88], [0, 3, 75, 110, 87], [0, 3, 72, 75, 110], [0, 3, 71, 75, 110], [64, 0, 3, 75, 110], [0, 3, 75, 110, 63], [0, 3, 75, 110, 62], [0, 3, 75, 110, 61], [0, 3, 75, 110, 60], [0, 3, 75, 110, 59], [0, 3, 75, 110, 58], [0, 3, 39, 75, 110], [0, 3, 38, 75, 110], [0, 3, 37, 75, 110], [0, 3, 36, 75, 110], [0, 3, 35, 75, 110], [0, 34, 3, 75, 110], [0, 3, 75, 110, 31], [0, 3, 75, 110, 30], [0, 3, 75, 110, 29], [0, 3, 75, 110, 25], [0, 3, 75, 110, 23], [0, 3, 75, 110, 21], [0, 3, 75, 110, 20], [0, 3, 75, 110, 16], [0, 3, 75, 13, 110], [0, 3, 75, 12, 110], [0, 3, 75, 11, 110], [0, 3, 10, 75, 110], [0, 3, 9, 75, 110], [0, 3, 8, 75, 110], [0, 3, 7, 75, 110], [129, 135, 79, 49, 115], [132, 138, 43, 122, 127], [96, 97, 50, 82, 94], [98, 67, 130, 75, 111], [4, 108, 44, 45, 114], [130, 100, 133, 83, 24], [102, 6, 113, 51, 57], [130, 41, 105, 75, 124], [97, 133, 82, 123, 94], [101, 137, 107, 121, 126], [134, 118, 90, 91, 93], [32, 128, 40, 137, 107], [1, 4, 73, 42, 55], [74, 77, 47, 116, 26], [134, 53, 118, 90, 124], [134, 44, 118, 90, 124], [96, 130, 133, 105, 82], [130, 82, 85, 89, 92], [134, 116, 118, 90, 93], [70, 6, 19, 54, 57], [130, 67, 105, 109, 95], [130, 75, 76, 117, 123], [33, 130, 105, 75, 84], [130, 133, 105, 82, 124], [0, 3, 99, 110, 81], [130, 109, 119, 89, 124], [134, 74, 77, 116, 26], [130, 74, 75, 78, 27], [128, 137, 107, 121, 126], [130, 134, 75, 80, 116], [130, 67, 134, 75, 111], [14, 15, 17, 18, 52, 22, 28], [66, 137, 107, 121, 126], [130, 133, 105, 83, 124], [134, 118, 119, 90, 124], [130, 6, 113, 54, 57], [130, 134, 75, 78, 27], [130, 133, 134, 116, 123], [134, 116, 86, 119, 124], [130, 104, 105, 75, 124], [0, 3, 75, 110, 111], [0, 130, 3, 75, 110], [68, 5, 73, 105, 112], [130, 105, 75, 119, 124], [65, 130, 105, 109, 119], [129, 135, 105, 79, 115], [131, 103, 136, 106, 125], [130, 105, 109, 119, 124], [132, 68, 138, 122, 127], [134, 109, 116, 119, 124], [1, 4, 73, 108, 55], [1, 4, 135, 115, 55], [129, 69, 105, 109, 119, 124], [136, 108, 114, 120, 125], [1, 131, 4, 135, 106, 108, 114, 55, 56], [137, 107, 114, 121, 126], [2, 46, 48, 122, 127]]
    K = kit.KitNET.load_kitnet("results/kitnet_model")
    normalize_params = torch.load('results/normalize_params.pt')
    mean = normalize_params['mean']
    std = normalize_params['std']

    # 创建 NormalizeTransform 对象
    normalize = NormalizeTransform(mean, std)
    gan_input_dim = len(feature_map)
    latent_dim = int(gan_input_dim * 0.5)

    generator = Generator(gan_input_dim, latent_dim)
    discriminator = Discriminator(gan_input_dim)
    encoder = Encoder(gan_input_dim, latent_dim)
    gsa = np.zeros([x_test.shape[0], len(feature_map)])
    for i in range(x_test.shape[0]):
        if i % 1000 == 0:
            print(i)
        gsa[i] = K.execute(x_test[i,]) #will train during the grace periods, then execute on all the rest.
    gsa = torch.from_numpy(gsa).float()
    y_test = torch.from_numpy(y_test)
    test_mnist = SimpleDataset(gsa, y_test,transform=normalize)
    test_dataloader = DataLoader(test_mnist, batch_size=1,shuffle=False)
    test_anomaly_detection(opt, generator, discriminator, encoder,
                        test_dataloader, device)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=30,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.9,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=8,
                        help="dimensionality of the latent space")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for "
                             "discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=3000,
                        help="interval betwen image samples")
    parser.add_argument("--split_rate", type=float, default=0.8,
                        help="rate of split for normal training data")
    parser.add_argument("--seed", type=int, default=42,
                        help="value of a random seed")
    opt = parser.parse_args()

    main(opt)
