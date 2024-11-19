import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model.gan import Generator, Discriminator, Encoder
import model.MAE as mae
import numpy as np
import pandas as pd
from model.MAEGAN import MAEGAN
import os
root_dir = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(root_dir)
from sklearn.ensemble import IsolationForest
import report_result
import read_data
import itertools
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# def load_cic2017_g():
#     seq_len = 5
#     embs_path = "/root/GCN/DyGCN/data/data/cic2017/model-DGC5-2.pt"
#     labels_path = "/root/GCN/DyGCN/data/data/cic2017/labels.npy"
#     train_len=[0, 527]
#     data_embs = torch.load(embs_path).detach().cpu().numpy()
#     x_train = data_embs[train_len[0]+seq_len:train_len[1]]
#     x_test=np.concatenate([data_embs[:train_len[0]],data_embs[train_len[1]:]])
#     y_train = np.zeros(len(x_train))
#     labels = np.load(labels_path, allow_pickle=True)
#     labels = labels.astype(int)
#     labels=labels[seq_len:]
#     y_test=np.concatenate((labels[:train_len[0]], labels[train_len[1]:]))
#     minmax_scaler = MinMaxScaler()
#     x_train = minmax_scaler.fit_transform(x_train)  # 仅在训练数据上拟合
#     x_test = minmax_scaler.transform(x_test)  # 使用相同的缩放器进行转换
#     return (x_train, y_train), (x_test, y_test)

# def load_ugr16_g():
#     seq_len = 5
#     embs_path = "/root/GCN/DyGCN/data/data/ugr16/model-DGC5-2.pt"
#     labels_path = "/root/GCN/DyGCN/data/data/ugr16/labels.npy"
#     train_len=[0, 500]
#     data_embs = torch.load(embs_path).detach().cpu().numpy()
#     x_train = data_embs[train_len[0]+seq_len:train_len[1]]
#     x_test=np.concatenate([data_embs[:train_len[0]],data_embs[train_len[1]:]])
#     y_train = np.zeros(len(x_train))
#     labels = np.load(labels_path, allow_pickle=True)
#     labels = labels.astype(int)
#     labels=labels[seq_len:]
#     y_test=np.concatenate((labels[:train_len[0]], labels[train_len[1]:]))
#     minmax_scaler = MinMaxScaler()
#     x_train = minmax_scaler.fit_transform(x_train)  # 仅在训练数据上拟合
#     x_test = minmax_scaler.fit_transform(x_test)  # 使用相同的缩放器进行转换
#     return (x_train, y_train), (x_test, y_test)

# def load_cic2017_o():
#     embs_path = "/root/GCN/DyGCN/data/data/cic2017/feats.npy"
#     labels_path = "/root/GCN/DyGCN/data/data/cic2017/label_types.npy"
#     data_embs = np.load(embs_path, allow_pickle=True).reshape(-1, 77)
#     labels = np.load(labels_path, allow_pickle=True).reshape(-1)
#     labels = np.where(labels == 'BENIGN', 0, 1)
#     x_train = data_embs[:527000]
#     y_train = labels[:527000]
#     x_test = data_embs[527000:]
#     y_test = labels[527000:]
#     minmax_scaler = MinMaxScaler()
#     x_train = minmax_scaler.fit_transform(x_train)  # 仅在训练数据上拟合
#     x_test = minmax_scaler.fit_transform(x_test)  # 使用相同的缩放器进行转换
#     return (x_train, y_train), (x_test, y_test)

# def load_UGR16():
#     raw_x_train = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)
#     x_train = raw_x_train
#     x_train = torch.from_numpy(x_train.values).float()
#     y_train = torch.zeros(len(x_train))


#     raw_x_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
#     x_test = raw_x_test
#     x_test = torch.from_numpy(x_test.values).float()
#     y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
#     y_test = torch.from_numpy(y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values)

#     # 接下来进行归一化
#     (x_train, y_train), (x_test, y_test) = (x_train.numpy(), y_train.numpy()), (x_test.numpy(), y_test.numpy())
#     minmax_scaler = MinMaxScaler()
#     x_train = minmax_scaler.fit_transform(x_train)  # 仅在训练数据上拟合
#     x_test = minmax_scaler.transform(x_test)  # 使用相同的缩放器进行转换
#     return (x_train, y_train), (x_test, y_test)

# def load_UGR16_faac():
#     df = pd.read_csv("/vdb2/FCParserSliding/example/data/ugr16.csv")
#     label_columns = ['label_dos', 'label_scan11', 'label_scan44', 'label_nerisbotnet']
#     label = (df[label_columns].sum(axis=1) > 0).astype(int)
#     features_to_drop = [
#         'timestamp', 'label_background', 'label_dos', 'label_scan44', 
#         'label_scan11', 'label_nerisbotnet', 'label_blacklist', 
#         'label_anomaly-udpscan', 'label_anomaly-sshscan', 
#         'label_anomaly-spam', 'label_other'
#     ]
#     f2 = ['dport_ftp_control', 'src_ip_public', 'dport_bootp', 'src_ip_default', 'sport_ftp_data', 'src_ip_private', 'dport_smtp', 'sport_ftp_control', 'sport_rapservice', 'dport_rapservice', 'dport_netbios', 'dport_ssh', 'dport_ntp', 'dport_mssql', 'dport_mysql', 'dport_smtp_ssl', 'dport_telnet', 'dport_http2', 'sport_bittorrent', 'sport_smtp_ssl', 'dst_ip_private', 'sport_ssh', 'sport_mssql', 'sport_http2', 'sport_snmp', 'protocol_icmp', 'dport_ftp_data', 'sport_mds', 'sport_mysql', 'dport_xmpp', 'sport_zero', 'sport_ntp', 'sport_bootp', 'nbytes_low', 'dport_snmp', 'sport_citrix', 'sport_smtp', 'sport_msnmessenger', 'dport_imap4.1', 'sport_oracle', 'npackets_veryhigh', 'tcpflags_URG', 'sport_mgc', 'sport_multiplex', 'dport_bittorrent', 'sport_imap4.1', 'dport_pop3', 'sport_pop3', 'sport_private', 'sport_smtp.1', 'protocol_other', 'sport_syslog', 'sport_netbios', 'sport_xmpp', 'dport_smtp.1', 'dport_mds', 'sport_metasploit', 'srctos_192', 'sport_emule', 'sport_socks', 'sport_telnet', 'dport_zero', 'sport_openvpn', 'dport_openvpn', 'sport_quote', 'sport_ldaps', 'sport_discard', 'sport_ldap', 'dport_quote', 'sport_daytime', 'sport_cups', 'dport_multiplex', 'srctos_other', 'sport_reserved', 'srctos_zero']

#     df = df.drop(columns=features_to_drop)
#     df = df.drop(columns=f2)
#     train_len=500
#     x_train = df.iloc[:train_len, :].to_numpy().astype(float)      # 前500行为训练集特征
#     y_train = label.iloc[:train_len].to_numpy()       # 前500行为训练集标签

#     x_test = df.iloc[train_len:, :].to_numpy().astype(float)        # 剩余部分为测试集特征
#     y_test = label.iloc[train_len:].to_numpy()        # 剩余部分为测试集标签
#     minmax_scaler = MinMaxScaler()
#     x_train = minmax_scaler.fit_transform(x_train)  # 仅在训练数据上拟合
#     x_test = minmax_scaler.transform(x_test)  # 使用相同的缩放器进行转换
#     return (x_train, y_train), (x_test, y_test)

# def load_cic2017_faac():
#     df = pd.read_csv("/vdb2/FCParserSliding/example/data/cic2017/cic2017.csv")
#     label_columns = ['label_DoS_Hulk', 'label_DDoS', 'label_PortScan', 'label_other']
#     label = (df[label_columns].sum(axis=1) > 0).astype(int)
#     features_to_drop = [
#         'timestamp', 'label_background', 'label_DoS_Hulk', 'label_DDoS', 'label_PortScan', 'label_other'
#     ]
#     print(label.value_counts())
#     df = df.drop(columns=features_to_drop)
#     train_len=527
#     x_train = df.iloc[:train_len, :].to_numpy()      # 前500行为训练集特征
#     y_train = label.iloc[:train_len].to_numpy()       # 前500行为训练集标签

#     x_test = df.iloc[train_len:, :].to_numpy()        # 剩余部分为测试集特征
#     y_test = label.iloc[train_len:].to_numpy()        # 剩余部分为测试集标签
#     minmax_scaler = MinMaxScaler()
#     x_train = minmax_scaler.fit_transform(x_train)  # 仅在训练数据上拟合
#     x_test = minmax_scaler.transform(x_test)  # 使用相同的缩放器进行转换
#     return (x_train, y_train), (x_test, y_test)


# def load_cic2018_faac():
#     df = pd.read_csv("/vdb2/FCParserSliding/example/data/cic2018/cic2018.csv")
#     label_columns = ['label_other']
#     label = (df[label_columns].sum(axis=1) > 0).astype(int)
#     features_to_drop = [
#         'timestamp', 'label_background', 'label_other'
#     ]
#     print(label.value_counts())
#     df = df.drop(columns=features_to_drop)
#     train_len=4600
#     x_train = df.iloc[:train_len, :].to_numpy()      # 前500行为训练集特征
#     y_train = label.iloc[:train_len].to_numpy()       # 前500行为训练集标签

#     x_test = df.iloc[train_len:, :].to_numpy()        # 剩余部分为测试集特征
#     y_test = label.iloc[train_len:].to_numpy()        # 剩余部分为测试集标签
#     minmax_scaler = MinMaxScaler()
#     x_train = minmax_scaler.fit_transform(x_train)  # 仅在训练数据上拟合
#     x_test = minmax_scaler.transform(x_test)  # 使用相同的缩放器进行转换
#     return (x_train, y_train), (x_test, y_test)

def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # (x_train, y_train), (x_test, y_test) = load_UNSW()
    if opt.dataset == "ugr16":
        (x_train, y_train), (x_test, y_test) = read_data.load_UGR16_faac()
        filepath = "load_UGR16_faac"
        maegan = MAEGAN(opt, input_dim = x_train.shape[1], maxAE=10, minAE=1, filepath=filepath, batch_size=4)
        # maegan = MAEGAN(opt, input_dim = x_train.shape[1], maxAE=1, minAE=1, filepath=filepath, batch_size=4)
        # model = "f-anogan"
        # opt.n_epochs = 14
        # maegan = MAEGAN(opt, input_dim = x_train.shape[1], maxAE=1, minAE=1, filepath=filepath, batch_size=3)
    elif opt.dataset == "cic2017":
        model = "MAEGAN"
        (x_train, y_train), (x_test, y_test) = read_data.load_cic2017_faac()
        filepath = "load_cic2017_faac"
        # maegan = MAEGAN(opt, input_dim = x_train.shape[1], maxAE=10, minAE=1, filepath=filepath, batch_size=4)
        maegan = MAEGAN(opt, input_dim = x_train.shape[1], maxAE=10, minAE=1, filepath=filepath, batch_size=8)
    elif opt.dataset == "cic2018":
        model = "MAEGAN"
        (x_train, y_train), (x_test, y_test) = read_data.load_cic2018_faac()
        filepath = "load_cic2018_faac"
        maegan = MAEGAN(opt, input_dim = x_train.shape[1], maxAE=10, minAE=1, filepath=filepath, batch_size=4)
        # model = "f-anogan"
        # maegan = MAEGAN(opt, input_dim = x_train.shape[1], maxAE=1, minAE=1, filepath=filepath, batch_size=4)

    print("Running KitNET:")
    maegan.train(x_train)
    score, _ = maegan.test(x_test, y_test)
    report_result.report_result(model=model, name=filepath, anomaly_score=score, labels=y_test)

def main_for_gcn(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # (x_train, y_train), (x_test, y_test) = load_UNSW()
    if opt.dataset == "ugr16v2":
        model = "MAEGAN"
        (x_train, y_train), (x_test, y_test) = read_data.load_UGR16_DyGAT()
        filepath = "load_UGR16_DyGAT"
        maegan = MAEGAN(opt, input_dim = x_train.shape[1], maxAE=10, minAE=1, filepath=filepath, batch_size=4)
    elif opt.dataset == "ugr16v2-1":
        model = "MAEGAN"
        (x_train, y_train), (x_test, y_test) = read_data.load_UGR16_DyGAT_withoutFA()
        filepath = "load_UGR16_DyGAT_withoutFA"
        maegan = MAEGAN(opt, input_dim = x_train.shape[1], maxAE=10, minAE=1, filepath=filepath, batch_size=4)
    elif opt.dataset == "ugr16v2-2":
        model = "MAEGAN"
        (x_train, y_train), (x_test, y_test) = read_data.load_UGR16_DyGAT_withoutDY()
        filepath = "load_UGR16_DyGAT_withoutDY"
        maegan = MAEGAN(opt, input_dim = x_train.shape[1], maxAE=10, minAE=1, filepath=filepath, batch_size=4)
    elif opt.dataset == "cic2018v2":
        model = "MAEGAN"
        (x_train, y_train), (x_test, y_test) = read_data.load_CIC2018_DyGAT()
        filepath = "load_cic2018_DyGAT"
        maegan = MAEGAN(opt, input_dim = x_train.shape[1], maxAE=8, minAE=1, filepath=filepath, batch_size=16)
    elif opt.dataset == "cic2018v2-1":
        model = "MAEGAN"
        (x_train, y_train), (x_test, y_test) = read_data.load_UGR16_DyGAT_withoutFA()
        filepath = "load_cic2018_DyGAT_withoutFA"
        maegan = MAEGAN(opt, input_dim = x_train.shape[1], maxAE=4, minAE=1, filepath=filepath, batch_size=16)
    elif opt.dataset == "cic2018v2-2":
        model = "MAEGAN"
        (x_train, y_train), (x_test, y_test) = read_data.load_CIC2018_DyGAT_withoutDY()
        filepath = "load_cic2018_DyGAT_withoutDY"
        maegan = MAEGAN(opt, input_dim = x_train.shape[1], maxAE=8, minAE=1, filepath=filepath, batch_size=16)

    print("Running KitNET:")
    maegan.train(x_train)
    score, _ = maegan.test(x_test, y_test)
    report_result.report_result(model=model, name=filepath, anomaly_score=score, labels=y_test)

def main_itertools_for_gcn(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    # (x_train, y_train), (x_test, y_test) = read_data.load_cic2017_faac()
    # filepath = "load_cic2017_faac"
    # (x_train, y_train), (x_test, y_test) = read_data.load_cic2018_faac()
    # filepath = "load_cic2018_faac"
    (x_train, y_train), (x_test, y_test) = read_data.load_CIC2018_DyGAT()
    filepath = "load_cic2018_DyGAT"
    model = "MAEGAN"
    
    # 定义输出文件夹和文件路径
    output_dir = f"/root/bishe/results/{filepath}/MAEGAN/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "itertoolsresults.txt")
    
    # 定义参数范围
    maxAE_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    minAE_values = [1]
    batch_sizes = [4, 8, 16]
    # 初始化结果存储
    best_params = None
    best_score = 0
    results = []
    
    # 打开文件准备写入
    with open(output_file, "w") as f:
        # 遍历所有参数组合
        for batch_size, maxAE, minAE in itertools.product(batch_sizes, maxAE_values, minAE_values):
            if type(opt.seed) is int:
                torch.manual_seed(opt.seed)
            print(f"Testing with maxAE={maxAE}, minAE={minAE}, batch_size={batch_size}")
            if minAE > maxAE:
                continue
            
            # 初始化模型
            maegan = MAEGAN(opt, input_dim=x_train.shape[1], maxAE=maxAE, minAE=minAE, filepath=filepath, batch_size=batch_size)
            
            # 训练模型
            train_time = maegan.train(x_train)
            
            # 测试模型
            score, test_time = maegan.test(x_test, y_test)
            
            # 计算评估指标
            f1, result_str = report_result.report_result(model=model, name=filepath, anomaly_score=score, labels=y_test)
            
            # 存储结果
            results.append((maxAE, minAE, batch_size, f1))
            
            # 写入当前组合结果到文件
            f.write(f"maxAE={maxAE}, minAE={minAE}, batch_size={batch_size}, f1={f1:.4f}\n")
            f.write(f"{result_str}\n")
            f.write(f"train_time={train_time}, test_time={test_time}\n")
            
            # 更新最佳参数
            if f1 > best_score:
                best_score = f1
                best_params = (maxAE, minAE, batch_size)
        
        # 写入最佳参数到文件
        f.write("\nBest parameters found:\n")
        f.write(f"maxAE={best_params[0]}, minAE={best_params[1]}, batch_size={best_params[2]}, best accuracy={best_score:.4f}\n")
    
    print("\nBest parameters found:")
    print(f"maxAE={best_params[0]}, minAE={best_params[1]}, batch_size={best_params[2]}")
    print(f"Best accuracy: {best_score}")

    return results, best_params, best_score

# def main_itertools(opt):
#     if type(opt.seed) is int:
#         torch.manual_seed(opt.seed)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     # 加载数据
#     # (x_train, y_train), (x_test, y_test) = read_data.load_cic2017_faac()
#     # filepath = "load_cic2017_faac"
#     # (x_train, y_train), (x_test, y_test) = read_data.load_cic2018_faac()
#     # filepath = "load_cic2018_faac"
#     (x_train, y_train), (x_test, y_test) = read_data.load_UGR16_faac()
#     filepath = "load_UGR16_faac"
#     model = "MAEGAN"
    
#     # 定义输出文件夹和文件路径
#     output_dir = f"/root/bishe/results/{filepath}/MAEGAN/"
#     os.makedirs(output_dir, exist_ok=True)
#     output_file = os.path.join(output_dir, "itertoolsresults.txt")
    
#     # 定义参数范围
#     maxAE_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
#     minAE_values = [1]
#     batch_sizes = [4, 8, 16]
#     # 初始化结果存储
#     best_params = None
#     best_score = 0
#     results = []
    
#     # 打开文件准备写入
#     with open(output_file, "w") as f:
#         # 遍历所有参数组合
#         for batch_size, maxAE, minAE in itertools.product(batch_sizes, maxAE_values, minAE_values):
#             if type(opt.seed) is int:
#                 torch.manual_seed(opt.seed)
#             print(f"Testing with maxAE={maxAE}, minAE={minAE}, batch_size={batch_size}")
#             if minAE > maxAE:
#                 continue
            
#             # 初始化模型
#             maegan = MAEGAN(opt, input_dim=x_train.shape[1], maxAE=maxAE, minAE=minAE, filepath=filepath, batch_size=batch_size)
            
#             # 训练模型
#             train_time = maegan.train(x_train)
            
#             # 测试模型
#             score, test_time = maegan.test(x_test, y_test)
            
#             # 计算评估指标
#             f1, result_str = report_result.report_result(model=model, name=filepath, anomaly_score=score, labels=y_test)
            
#             # 存储结果
#             results.append((maxAE, minAE, batch_size, f1))
            
#             # 写入当前组合结果到文件
#             f.write(f"maxAE={maxAE}, minAE={minAE}, batch_size={batch_size}, f1={f1:.4f}\n")
#             f.write(f"{result_str}\n")
#             f.write(f"train_time={train_time}, test_time={test_time}\n")
            
#             # 更新最佳参数
#             if f1 > best_score:
#                 best_score = f1
#                 best_params = (maxAE, minAE, batch_size)
        
#         # 写入最佳参数到文件
#         f.write("\nBest parameters found:\n")
#         f.write(f"maxAE={best_params[0]}, minAE={best_params[1]}, batch_size={best_params[2]}, best accuracy={best_score:.4f}\n")
    
#     print("\nBest parameters found:")
#     print(f"maxAE={best_params[0]}, minAE={best_params[1]}, batch_size={best_params[2]}")
#     print(f"Best accuracy: {best_score}")

#     return results, best_params, best_score



"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.99,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.99,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for "
                             "discriminator per iter")
    parser.add_argument("--seed", type=int, default=42,
                        help="value of a random seed")
    parser.add_argument("--dataset", type=str, default="ugr16",
                        help="value of a random seed")
    opt = parser.parse_args()

    # main_itertools(opt)
    # main(opt)
    main_itertools_for_gcn(opt)

