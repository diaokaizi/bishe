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

def load_cic2017_faac():
    df = pd.read_csv("data/cic2017.csv")
    label_columns = ['label_DoS_Hulk', 'label_DDoS', 'label_PortScan', 'label_other']
    label = (df[label_columns].sum(axis=1) > 0).astype(int)
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
    train_len=4600
    x_train = df.iloc[:train_len, :].to_numpy().astype('float32')      # 前500行为训练集特征
    y_train = label.iloc[:train_len].to_numpy()       # 前500行为训练集标签

    x_test = df.iloc[train_len:, :].to_numpy().astype('float32')        # 剩余部分为测试集特征
    y_test = label.iloc[train_len:].to_numpy()        # 剩余部分为测试集标签
    minmax_scaler = MinMaxScaler()
    x_train = minmax_scaler.fit_transform(x_train)  # 仅在训练数据上拟合
    x_test = minmax_scaler.transform(x_test)  # 使用相同的缩放器进行转换
    return (x_train, y_train), (x_test, y_test)