import KitNET as kit
import numpy as np
import pandas as pd
import time

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
train = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtrain.csv").drop(columns=['Row'], axis=1).values
test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv").drop(columns=['Row'], axis=1).values
# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = int(train.shape[0] * 0.1) #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = train.shape[0] - FMgrace #the number of instances used to train the anomaly detector (ensemble itself)
print(FMgrace, ADgrace)
# Build KitNET
K = kit.KitNET(train.shape[1],maxAE,FMgrace,ADgrace)

print("Running KitNET:")
start = time.time()
# Here we process (train/execute) each individual observation.
# In this way, X is essentially a stream, and each observation is discarded after performing process() method.
for i in range(train.shape[0]):
    if i % 1000 == 0:
        print(i)
    K.process(train[i,]) #will train during the grace periods, then execute on all the rest.

