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
print(os.getcwd())
from sklearn.ensemble import IsolationForest
import report_result
import read_data
import importlib as imp
import time
from itertools import product

# 准备数据

model_name = "DeepSVDD"

# config: {'lr': 0.0005, 'hidden_dims': 3, 'epochs': 150}
# f1: 0.6563467492260062, traintime: 1.02, testtime:1.02  str: auc_score:0.7838,acc:0.8148,pre0.7626,rec:0.5761, f1:0.6563
# (x_train, y_train), (x_test, y_test) = read_data.load_cic2017_faac()
# filepath = "load_cic2017_faac"
# (x_train, y_train), (x_test, y_test) = read_data.load_cic2018_faac()
# filepath = "load_cic2018_faac"


module = imp.import_module('deepod.models.tabular')
model_class = getattr(module, "DeepSVDD")
# (x_train, y_train), (x_test, y_test) = read_data.load_UGR16_faac()
# filepath = "load_UGR16_faac"
# lr_values = [0.0005]
# hidden_dims_values = [3]
# epochs_values = [135]
(x_train, y_train), (x_test, y_test) = read_data.load_cic2018_faac()
filepath = "load_cic2018_faac"
lr_values = [1e-05]
hidden_dims_values = [100]
epochs_values = [64]
# config: {'lr': 1e-05, 'hidden_dims': 100, 'epochs': 60}
# f1: 0.8679475449832266, traintime: 4.55, testtime:4.55  str: auc_score:0.9012,acc:0.8683,pre0.7972,rec:0.9525, f1:0.8679
hyperparameter_grid = list(product(lr_values, hidden_dims_values, epochs_values))
for config_idx, (lr, hidden_dims, epochs) in enumerate(hyperparameter_grid):
    model_configs = {
        'lr': lr,
        'hidden_dims': hidden_dims,
        'epochs': epochs
    }
    clf = model_class(**model_configs, random_state=42, device="cpu")


    start_time = time.time()
    iof=clf.fit(x_train)
    end_time = time.time()
    epoch_duration = (end_time - start_time)  # 计算该 epoch 的时间并转换为毫秒
    print(f'Average Training Time per Epoch: {epoch_duration:.2f} ms')
    score = clf.decision_function(x_test)
    testing_time_cost = end_time - start_time
    print(f"Testing Time Cost: {testing_time_cost:.2f} seconds")
    f1, str = report_result.report_result(model=model_name, name=filepath, anomaly_score=score, labels=y_test)
    with open("/root/bishe/compare_models/DeepSVDD/cic2018", 'a') as f:
        f.write(f"config: {model_configs}\n")
        f.write(f"f1: {f1}, traintime: {epoch_duration:.2f}, testtime:{testing_time_cost:.2f}  str: {str}\n")
    



# UGR16

# acc:0.7957,pre0.6658,rec:0.6712, f1:0.6685Config 3: UNSW, 0.7980, 0.0000, 0.7578, 0.0000, 0.6630, 0.0000, 1501.9/1510.3s, DeepSVDD, {'lr': 0.01, 'hidden_dims': 64, 'epochs': 200}


# acc:0.8057,pre0.6762,rec:0.7038, f1:0.6897Config 1: UNSW, 0.8577, 0.0000, 0.7679, 0.0000, 0.6875, 0.0000, 7159.7/7510.3s, SLAD, {'lr': 1e-05, 'hidden_dims': 80, 'epochs': 40}



# CIC2017
# auc_score: 0.7705588336061516
# acc:0.7943,pre0.8787,rec:0.6578, f1:0.7524Config 1: cic2017, 0.7706, 0.0000, 0.7562, 0.0000, 0.6220, 0.0000, 2518.3/2533.9s, DeepSVDD, {'lr': 0.1, 'hidden_dims': 100, 'epochs': 200}

# auc_score: 0.8168864440976055
# acc:0.7747,pre0.8795,rec:0.6092, f1:0.7198Config 1: cic2017, 0.8169, 0.0000, 0.8278, 0.0000, 0.7303, 0.0000, 4286.4/5000.5s, SLAD, {'lr': 1e-05, 'hidden_dims': 80, 'epochs': 20}
# 
# 
# CIC2018
# auc_score: 0.9062172552792414
# acc:0.8428,pre0.7842,rec:0.9023, f1:0.8391Config 1: cic2018, 0.9062, 0.0000, 0.8234, 0.0000, 0.8166, 0.0000, 8081.4/8106.1s, DeepSVDD, {'lr': 0.005, 'hidden_dims': 70, 'epochs': 100}
# auc_score: 0.9054101270448554
# acc:0.8741,pre0.8086,rec:0.9471, f1:0.8724Config 1: cic2018, 0.9054, 0.0000, 0.7805, 0.0000, 0.8387, 0.0000, 21290.0/22479.0s, SLAD, {'lr': 0.0001, 'hidden_dims': 80, 'epochs': 50}
