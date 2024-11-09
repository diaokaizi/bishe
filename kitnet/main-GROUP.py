import sys
sys.path.append('../..')
import KitNET as kit
import numpy as np
import pandas as pd
import time
import torch

##############################################################################
# KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders.
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection

# This script demonstrates KitNET's ability to incrementally learn, and detect anomalies.
# The demo involves an m-by-n dataset with n=115 dimensions (features), and m=100,000 observations.
# Each observation is a snapshot of the network's state in terms of incremental damped statistics (see the NDSS paper for more details)

#The runtimes presented in the paper, are based on the C++ implimentation (roughly 100x faster than the python implimentation)
###################  Last Tested with Anaconda 2.7.14   #######################

# Load sample dataset (a recording of the Mirai botnet malware being activated)
# The first 70,000 observations are clean...


print("Reading Sample dataset...")
# X = pd.read_csv("/root/KitNET-py/UGR16/UGR16v1.Xtrain.csv").values #an m-by-n dataset with m observations
# KitNET params:
def load_group_np():
    seq_len=5
    embs_path = "/root/GCN/DyGCN/data/data/ugr16/graph_embs333.pt"
    # embs_path='data/graph_embs.pt'
    labels_path = "/root/GCN/DyGCN/data/data/ugr16/labels.npy"

    train_len=[0, 500]

    data_embs = torch.load(embs_path).detach().cpu().numpy()
    print(len(data_embs))
    print(data_embs.shape)
    print(data_embs[train_len[0]+seq_len:train_len[1]].shape)
    print(np.concatenate([data_embs[:train_len[0]],data_embs[train_len[1]:]]).shape)
    labels = np.load(labels_path, allow_pickle=True)
    labels=labels[seq_len:]
    labels=np.concatenate((labels[:train_len[0]], labels[train_len[1]:]))
    print(labels)
    x_train = data_embs[train_len[0]+seq_len:train_len[1]]
    test_embs=np.concatenate([data_embs[:train_len[0]],data_embs[train_len[1]:]])
    x_test = test_embs
    y_test = labels
    y_train = np.zeros(len(x_train))
    return (x_train, y_train), (x_test, y_test)

(train, y_train), (test, y_test) = load_group_np()


maxAE = 10 #maximum size for any autoencoder in the ensemble layer
# Build KitNET
K = kit.KitNET(train.shape[1],maxAE,train.shape[0],train.shape[0])

print("Running KitNET:")
start = time.time()
# Here we process (train/execute) each individual observation.
# In this way, X is essentially a stream, and each observation is discarded after performing process() method.
for i in range(train.shape[0]):
    if i % 1000 == 0:
        print(i)
    K.process(train[i,]) #will train during the grace periods, then execute on all the rest.
for i in range(train.shape[0]):
    if i % 1000 == 0:
        print(i)
    K.process(train[i,]) #will train during the grace periods, then execute on all the rest.

RMSEs = np.zeros(test.shape[0]) # a place to save the scores
for i in range(test.shape[0]):
    if i % 1000 == 0:
        print(i)
    RMSEs[i] = K.process(test[i,]) #will train during the grace periods, then execute on all the rest.
pd.DataFrame(RMSEs).to_csv("/root/bishe/kitnet/GROUP/RMSEs.csv", header=["remse"])
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))