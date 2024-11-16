import sys
sys.path.append('../..')
import KitNET as kit
import numpy as np
import pandas as pd
import time
import torch
import sys 
import os
root_dir = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(root_dir)
from sklearn.ensemble import IsolationForest
import report_result
import read_data
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


model_name = "MAEGAN-AE"

(x_train, y_train), (x_test, y_test) = read_data.load_cic2018_faac()
filepath = "load_cic2018_faac"
# (x_train, y_train), (x_test, y_test) = read_data.load_UGR16_faac()
# filepath = "load_UGR16_faac"
# (x_train, y_train), (x_test, y_test) = read_data.load_cic2017_faac()
# filepath = "load_cic2017_faac"

maxAE = 10 #maximum size for any autoencoder in the ensemble layer
# Build KitNET
K = kit.KitNET(x_train.shape[1], 10, 1,x_train.shape[0],x_train.shape[0])

print("Running KitNET:")
start = time.time()
# Here we process (train/execute) each individual observation.
# In this way, X is essentially a stream, and each observation is discarded after performing process() method.
for i in range(x_train.shape[0]):
    if i % 1000 == 0:
        print("fa", i)
    K.process(x_train[i,]) #will train during the grace periods, then execute on all the rest.
for i in range(x_train.shape[0]):
    if i % 1000 == 0:
        print("train", i)
    K.process(x_train[i,]) #will train during the grace periods, then execute on all the rest.

RMSEs = np.zeros(x_test.shape[0]) # a place to save the scores
for i in range(x_test.shape[0]):
    if i % 1000 == 0:
        print("test", i)
    RMSEs[i] = K.process(x_test[i,]) #will train during the grace periods, then execute on all the rest.
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))
report_result.report_result(model=model_name, name=filepath, anomaly_score=RMSEs, labels=y_test)