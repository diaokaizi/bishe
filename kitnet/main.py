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

RMSEs = np.zeros(test.shape[0]) # a place to save the scores
for i in range(test.shape[0]):
    if i % 1000 == 0:
        print(i)
    RMSEs[i] = K.process(test[i,]) #will train during the grace periods, then execute on all the rest.
pd.DataFrame(RMSEs).to_csv("RMSEs.csv", header=["remse"])
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))
# Here we demonstrate how one can fit the RMSE scores to a log-normal distribution (useful for finding/setting a cutoff threshold \phi)
from scipy.stats import norm
benignSample = np.log(RMSEs)
logProbs = norm.logsf(np.log(RMSEs), np.mean(benignSample), np.std(benignSample))

# plot the RMSE anomaly scores
print("Plotting results")
from matplotlib import pyplot as plt
from matplotlib import cm
timestamps = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv")['Row'].values
# fig = plt.scatter(timestamps[FMgrace+ADgrace+1:],RMSEs[FMgrace+ADgrace+1:],s=0.1,c=logProbs[FMgrace+ADgrace+1:],cmap='RdYlGn')
fig = plt.scatter(timestamps,RMSEs)
plt.title("Anomaly Scores from KitNET's Execution Phase")
plt.ylabel("RMSE (log scaled)")
plt.xlabel("Time elapsed [min]")
# plt.annotate('Mirai C&C channel opened [Telnet]', xy=(timestamps[71662],RMSEs[71662]), xytext=(timestamps[58000],1),arrowprops=dict(facecolor='black', shrink=0.05),)
# plt.annotate('Mirai Bot Activated\nMirai scans network for vulnerable devices', xy=(timestamps[72662],1), xytext=(timestamps[55000],5),arrowprops=dict(facecolor='black', shrink=0.05),)
figbar=plt.colorbar()
# plt.show()
plt.savefig("UGR16.png")
