import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd

def load_UGR16_faac():
    df = pd.read_csv("data/ugr16.csv")
    label_columns = ['label_dos', 'label_scan11', 'label_scan44', 'label_nerisbotnet']
    label = (df[label_columns].sum(axis=1) > 0).astype(int)
    features_to_drop = [
        'timestamp', 'label_background', 'label_dos', 'label_scan44', 
        'label_scan11', 'label_nerisbotnet', 'label_blacklist', 
        'label_anomaly-udpscan', 'label_anomaly-sshscan', 
        'label_anomaly-spam', 'label_other'
    ]
    f2 = ['dport_ftp_control', 'src_ip_public', 'dport_bootp', 'src_ip_default', 'sport_ftp_data', 'src_ip_private', 'dport_smtp', 'sport_ftp_control', 'sport_rapservice',
          'dport_rapservice', 'dport_netbios', 'dport_ssh', 'dport_ntp', 'dport_mssql', 'dport_mysql', 'dport_smtp_ssl', 'dport_telnet', 'dport_http2', 'sport_bittorrent',
          'sport_smtp_ssl', 'dst_ip_private', 'sport_ssh', 'sport_mssql', 'sport_http2', 'sport_snmp', 'protocol_icmp', 'dport_ftp_data', 'sport_mds', 'sport_mysql', 'dport_xmpp',
          'sport_zero', 'sport_ntp', 'sport_bootp', 'nbytes_low', 'dport_snmp', 'sport_citrix', 'sport_smtp', 'sport_msnmessenger', 'dport_imap4.1', 'sport_oracle', 'npackets_veryhigh',
          'tcpflags_URG', 'sport_mgc', 'sport_multiplex', 'dport_bittorrent', 'sport_imap4.1', 'dport_pop3', 'sport_pop3', 'sport_private', 'sport_smtp.1', 'protocol_other', 'sport_syslog',
          'sport_netbios', 'sport_xmpp', 'dport_smtp.1', 'dport_mds', 'sport_metasploit', 'srctos_192', 'sport_emule', 'sport_socks', 'sport_telnet', 'dport_zero', 'sport_openvpn', 'dport_openvpn',
          'sport_quote', 'sport_ldaps', 'sport_discard', 'sport_ldap', 'dport_quote', 'sport_daytime', 'sport_cups', 'dport_multiplex', 'srctos_other', 'sport_reserved', 'srctos_zero',
          'protocol_igmp', 'tcpflags_ACK', 'tcpflags_PSH', 'tcpflags_RST', 'tcpflags_SYN', 'tcpflags_FIN']

    df = df.drop(columns=features_to_drop)
    df = df.drop(columns=f2)
    train_len=500
    x_train = df.iloc[:train_len, :].to_numpy().astype('float32')      # 前500行为训练集特征
    y_train = label.iloc[:train_len].to_numpy()       # 前500行为训练集标签

    x_test = df.iloc[train_len:, :].to_numpy().astype('float32')        # 剩余部分为测试集特征
    y_test = label.iloc[train_len:].to_numpy()        # 剩余部分为测试集标签
    minmax_scaler = MinMaxScaler()
    x_train = minmax_scaler.fit_transform(x_train)  # 仅在训练数据上拟合
    x_test = minmax_scaler.transform(x_test)  # 使用相同的缩放器进行转换
    return (x_train, y_train), (x_test, y_test)


def load_UGR16_DyGAT():
    seq_len = 5
    data = torch.load("/vdb2/GCN/DyGCN/data/data/ugr16/model-DGC5.pt").detach().cpu().numpy()
    labels = np.load("/vdb2/GCN/DyGCN/data/data/ugr16/labels.npy", allow_pickle=True)
    train_len=500
    x_train = data[:train_len]     # 前500行为训练集特征
    x_test = data[train_len:]       # 前500行为训练集标签
    y_train=labels[:train_len]
    y_test=labels[train_len + seq_len:]
    return (x_train, y_train), (x_test, y_test)

def load_UGR16_DyGAT_withoutFA():
    seq_len = 5
    data = torch.load("/vdb2/GCN/DyGCN/data/data/ugr16/model-DGC5-withoutFA.pt").detach().cpu().numpy()
    labels = np.load("/vdb2/GCN/DyGCN/data/data/ugr16/labels.npy", allow_pickle=True)
    train_len=500
    x_train = data[:train_len]     # 前500行为训练集特征
    x_test = data[train_len:]       # 前500行为训练集标签
    y_train=labels[:train_len]
    y_test=labels[train_len + seq_len:]
    return (x_train, y_train), (x_test, y_test)

def load_UGR16_DyGAT_withoutDY():
    seq_len = 5
    data = torch.load("/vdb2/GCN/DyGCN/data/data/ugr16/model-DGC5-withoutDY.pt").detach().cpu().numpy()
    labels = np.load("/vdb2/GCN/DyGCN/data/data/ugr16/labels.npy", allow_pickle=True)
    train_len=500
    x_train = data[:train_len]     # 前500行为训练集特征
    x_test = data[train_len:]       # 前500行为训练集标签
    y_train=labels[:train_len]
    y_test=labels[train_len + seq_len:]
    return (x_train, y_train), (x_test, y_test)

def load_CIC2018_DyGAT():
    seq_len = 5
    data = torch.load("/vdb2/GCN/DyGCN/data/data/cic2018/model-DGC5.pt").detach().cpu().numpy()
    labels = np.load("/vdb2/GCN/DyGCN/data/data/cic2018/labels.npy", allow_pickle=True)
    train_len=4600
    x_train = data[:train_len]     # 前500行为训练集特征
    x_test = data[train_len:]       # 前500行为训练集标签
    y_train=labels[:train_len]
    y_test=labels[train_len + seq_len:]
    return (x_train, y_train), (x_test, y_test)

def load_CIC2018_DyGAT_withoutFA():
    seq_len = 5
    data = torch.load("/vdb2/GCN/DyGCN/data/data/cic2018/model-DGC5-withoutFA.pt").detach().cpu().numpy()
    labels = np.load("/vdb2/GCN/DyGCN/data/data/cic2018/labels.npy", allow_pickle=True)
    train_len=4600
    x_train = data[:train_len]     # 前500行为训练集特征
    x_test = data[train_len:]       # 前500行为训练集标签
    y_train=labels[:train_len]
    y_test=labels[train_len + seq_len:]
    return (x_train, y_train), (x_test, y_test)

def load_CIC2018_DyGAT_withoutDY():
    seq_len = 5
    data = torch.load("/vdb2/GCN/DyGCN/data/data/cic2018/model-DGC5-withoutDY.pt").detach().cpu().numpy()
    labels = np.load("/vdb2/GCN/DyGCN/data/data/cic2018/labels.npy", allow_pickle=True)
    train_len=4600
    x_train = data[:train_len]     # 前500行为训练集特征
    x_test = data[train_len:]       # 前500行为训练集标签
    y_train=labels[:train_len]
    y_test=labels[train_len + seq_len:]
    return (x_train, y_train), (x_test, y_test)



def load_cic2017_faac():
    df = pd.read_csv("data/cic2017.csv")
    # label_background,label_DoS_Hulk,label_DDoS,label_PortScan,label_other
    label_columns = ['label_DoS_Hulk', 'label_DDoS', 'label_PortScan', 'label_other']
    # label = (df[label_columns].sum(axis=1) != 1000).astype(int)
    label = df['label_background'].apply(lambda x: 0 if x == 1000 else 1)
    features_to_drop = [
        'timestamp', 'label_background', 'label_DoS_Hulk', 'label_DDoS', 'label_PortScan', 'label_other'
    ]
    print(label.value_counts())
    df = df.drop(columns=features_to_drop)
    # f2 = ['label_background', 'label_DoS_Hulk', 'label_DDoS', 'label_PortScan', 'label_other', 'dst_ip_private', 'dst_ip_public', 'dport_http', 'dport_https', 'dport_ftp_control', 'dport_ssh', 'sport_ftp_control', 'nbytes_b_high', 'nbytes_b_medium', 'sport_https', 'dport_http2', 'src_ip_public', 'src_ip_private', 'nbytes_b_low', 'dport_dns', 'protocol_6', 'protocol_17', 'sport_http2', 'dport_private', 'sport_register', 'nbytes_medium', 'sport_http', 'dport_ntp', 'sport_ntp', 'npackets_low', 'sport_reserved', 'nbytes_b_veryhigh', 'npackets_b_low', 'sport_zero', 'dport_zero', 'protocol_other', 'npackets_b_high', 'dport_register', 'npackets_b_verylow', 'npackets_verylow', 'sport_private', 'npackets_medium', 'nbytes_verylow', 'dport_chwhereen', 'dport_cups']
    # f2 = ['dst_ip_private', 'dst_ip_public', 'dport_http', 'dport_https', 'dport_ftp_control', 'dport_ssh', 'sport_ftp_control', 'nbytes_b_high', 'nbytes_b_medium', 'sport_https', 'dport_http2', 'src_ip_public', 'src_ip_private', 'nbytes_b_low', 'dport_dns', 'protocol_6', 'protocol_17', 'sport_http2', 'dport_private', 'sport_register']
    # df = df[f2]
    print(df.shape)
    train_len=529
    x_train = df.iloc[:train_len, :].to_numpy().astype('float32')      # 前500行为训练集特征
    y_train = label.iloc[:train_len].to_numpy()       # 前500行为训练集标签

    # x_train = x_train[y_train != 1]
    # y_train = y_train[y_train != 1]
    print(x_train.shape)


    x_test = df.iloc[train_len:, :].to_numpy().astype('float32')        # 剩余部分为测试集特征
    y_test = label.iloc[train_len:].to_numpy()        # 剩余部分为测试集标签
    minmax_scaler = MinMaxScaler()
    x_train = minmax_scaler.fit_transform(x_train)  # 仅在训练数据上拟合
    x_test = minmax_scaler.transform(x_test)  # 使用相同的缩放器进行转换
    return (x_train, y_train), (x_test, y_test)


def load_cic2018_faac():
    df = pd.read_csv("data/cic2018.csv")
    label_columns = ['label_other']
    label = (df[label_columns].sum(axis=1) > 0).astype(int)
    features_to_drop = [
        'timestamp', 'label_background', 'label_other'
    ]
    print(label.value_counts())
    df = df.drop(columns=features_to_drop)

    # f2 = ['npackets_b_verylow', 'npackets_b_low', 'npackets_b_medium', 'npackets_b_high', 'npackets_b_veryhigh',
    #       'nbytes_b_verylow', 'nbytes_b_low','nbytes_b_medium','nbytes_b_high','nbytes_b_veryhigh']

    # f3 = ['npackets_verylow', 'npackets_low', 'npackets_medium', 'npackets_high', 'npackets_veryhigh',
    #       'nbytes_verylow', 'nbytes_low','nbytes_medium','nbytes_high','nbytes_veryhigh']
    f4 = ['dst_ip_private', 'dst_ip_public', 'dport_https', 'protocol_6', 'protocol_17', 'dport_dns', 'nbytes_b_medium', 'nbytes_b_low', 'src_ip_public', 'src_ip_private', 'dport_http', 'sport_https', 'nbytes_b_high', 'nbytes_medium', 'npackets_medium', 'npackets_b_medium', 'npackets_b_low', 'sport_register', 'nbytes_b_veryhigh', 'npackets_low', 'src_ip_default', 'sport_daytime', 'sport_bootp', 'dport_bootp']
# npackets_verylow,npackets_low,npackets_medium,npackets_high,npackets_veryhigh,npackets_b_verylow,npackets_b_low,npackets_b_medium,npackets_b_high,npackets_b_veryhigh,nbytes_verylow,nbytes_low,nbytes_medium,nbytes_high,nbytes_veryhigh,nbytes_b_verylow,nbytes_b_low,nbytes_b_medium,nbytes_b_high,nbytes_b_veryhigh,label_background,label_other
# ,,,,,label_background,label_other
    df = df.drop(columns=f4)

    train_len=4600
    x_train = df.iloc[:train_len, :].to_numpy().astype('float32')      # 前500行为训练集特征
    y_train = label.iloc[:train_len].to_numpy()       # 前500行为训练集标签

    x_test = df.iloc[train_len:, :].to_numpy().astype('float32')        # 剩余部分为测试集特征
    y_test = label.iloc[train_len:].to_numpy()        # 剩余部分为测试集标签
    minmax_scaler = MinMaxScaler()
    x_train = minmax_scaler.fit_transform(x_train)  # 仅在训练数据上拟合
    x_test = minmax_scaler.transform(x_test)  # 使用相同的缩放器进行转换
    return (x_train, y_train), (x_test, y_test)

def load_UGR16_GCN_LSTM():
    seq_len = 5
    feat_list=np.load("/root/GCN/DyGCN/data/data/ugr16/feats.npy", allow_pickle=True)
    feat_list_mean = feat_list.mean(axis=(1))[seq_len:]
    data = torch.load("/root/bishe/gnn_result/GCN_LSTM/ugr16/graph_embs.pt").detach().cpu().numpy()[:,:10]
    data = np.concatenate([feat_list_mean, data], axis=1)
    labels = np.load("/root/bishe/gnn_result/GCN_LSTM/ugr16/labels.npy", allow_pickle=True)
    train_len=500
    x_train = data[:train_len]     # 前500行为训练集特征
    x_test = data[train_len:]       # 前500行为训练集标签
    y_train=labels[:train_len]
    y_test=labels[train_len + seq_len:]
    return (x_train, y_train), (x_test, y_test)

def load_UGR16_Evolve_GCN():
    seq_len = 5
    feat_list=np.load("/root/GCN/DyGCN/data/data/ugr16/feats.npy", allow_pickle=True)[seq_len:]
    data = torch.load("/root/bishe/gnn_result/Evolve_GCN/ugr16/graph_embs.pt").detach().cpu().numpy()
    feat_list_mean = feat_list.mean(axis=(1))
    # data = np.concatenate([feat_list_mean, data], axis=1)
    data = np.concatenate([feat_list_mean, data], axis=1)
    labels = np.load("/root/bishe/gnn_result/Evolve_GCN/ugr16/labels.npy", allow_pickle=True)
    train_len=500
    x_train = data[:train_len]     # 前500行为训练集特征
    x_test = data[train_len:]       # 前500行为训练集标签
    y_train=labels[:train_len]
    y_test=labels[train_len + seq_len:]
    return (x_train, y_train), (x_test, y_test)

def load_UGR16_GCN():
    feat_list=np.load("/root/GCN/DyGCN/data/data/ugr16/feats.npy", allow_pickle=True)
    data = torch.load("/root/bishe/gnn_result/GCN/ugr16/graph_embs.pt").detach().cpu().numpy()[:, :10]
    feat_list_mean = feat_list.mean(axis=(1))
    data = np.concatenate([feat_list_mean, data], axis=1)
    labels = np.load("/root/bishe/gnn_result/GCN/ugr16/labels.npy", allow_pickle=True)
    train_len=500
    x_train = data[:train_len]     # 前500行为训练集特征
    x_test = data[train_len:]       # 前500行为训练集标签
    y_train=labels[:train_len]
    y_test=labels[train_len:]
    return (x_train, y_train), (x_test, y_test)

def load_UGR16_GAT():
    data = torch.load("/root/bishe/gnn_result/GAT/ugr16/graph_embs.pt").detach().cpu().numpy()
    labels = np.load("/root/bishe/gnn_result/GAT/ugr16/labels.npy", allow_pickle=True)
    train_len=500
    x_train = data[:train_len]     # 前500行为训练集特征
    x_test = data[train_len:]       # 前500行为训练集标签
    y_train=labels[:train_len]
    y_test=labels[train_len:]
    return (x_train, y_train), (x_test, y_test)

def load_CIC2018_GCN_LSTM():
    seq_len = 5
    feat_list=np.load("/vdb2/GCN/DyGCN/data/data/cic2018/feats.npy", allow_pickle=True)[seq_len:]
    feat_list_mean = feat_list.mean(axis=(1))
    data = torch.load("/root/bishe/gnn_result/GCN_LSTM/cic2018/graph_embs.pt").detach().cpu().numpy()
    data = np.concatenate([feat_list_mean, data], axis=1)
    labels = np.load("/root/bishe/gnn_result/GCN_LSTM/cic2018/labels.npy", allow_pickle=True)
    train_len=4600
    x_train = data[:train_len]     # 前500行为训练集特征
    x_test = data[train_len:]       # 前500行为训练集标签
    y_train=labels[:train_len]
    y_test=labels[train_len + seq_len:]
    return (x_train, y_train), (x_test, y_test)

def load_CIC2018_Evolve_GCN():
    seq_len = 5
    feat_list=np.load("/vdb2/GCN/DyGCN/data/data/cic2018/feats.npy", allow_pickle=True)[seq_len:]
    feat_list_mean = feat_list.mean(axis=(1))
    data = torch.load("/root/bishe/gnn_result/Evolve_GCN/cic2018/graph_embs.pt").detach().cpu().numpy()
    data = np.concatenate([feat_list_mean, data], axis=1)
    labels = np.load("/root/bishe/gnn_result/Evolve_GCN/cic2018/labels.npy", allow_pickle=True)
    train_len=4600
    x_train = data[:train_len]     # 前500行为训练集特征
    x_test = data[train_len:]       # 前500行为训练集标签
    y_train=labels[:train_len]
    y_test=labels[train_len + seq_len:]
    return (x_train, y_train), (x_test, y_test)

def load_CIC2018_GCN():
    feat_list=np.load("/vdb2/GCN/DyGCN/data/data/cic2018/feats.npy", allow_pickle=True)
    feat_list_mean = feat_list.mean(axis=(1))[:, :50]
    data = torch.load("/root/bishe/gnn_result/GCN/cic2018/graph_embs.pt").detach().cpu().numpy()
    data = np.concatenate([feat_list_mean, data], axis=1)
    labels = np.load("/root/bishe/gnn_result/GCN/cic2018/labels.npy", allow_pickle=True)
    train_len=4600
    x_train = data[:train_len]     # 前500行为训练集特征
    x_test = data[train_len:]       # 前500行为训练集标签
    y_train=labels[:train_len]
    y_test=labels[train_len:]
    return (x_train, y_train), (x_test, y_test)

def load_CIC2018_GAT():
    feat_list=np.load("/vdb2/GCN/DyGCN/data/data/cic2018/feats.npy", allow_pickle=True)
    feat_list_mean = feat_list.mean(axis=(1))
    data = torch.load("/root/bishe/gnn_result/GAT/cic2018/graph_embs.pt").detach().cpu().numpy()
    data = np.concatenate([feat_list_mean, data], axis=1)
    labels = np.load("/root/bishe/gnn_result/GAT/cic2018/labels.npy", allow_pickle=True)
    train_len=4600
    x_train = data[:train_len]     # 前500行为训练集特征
    x_test = data[train_len:]       # 前500行为训练集标签
    y_train=labels[:train_len]
    y_test=labels[train_len:]
    return (x_train, y_train), (x_test, y_test)