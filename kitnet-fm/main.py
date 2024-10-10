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
# feature_map =  [[113], [34, 86], [64], [2, 5], [27, 79], [33, 85], [15, 67], [101], [42, 94], [0, 3], [60], [47, 99], [37, 89], [51], [14, 66], [44, 96], [63], [11, 12], [46, 98], [13, 65], [77], [53, 105, 50, 102], [78], [90], [48, 100], [45, 97], [26], [122], [49, 36, 88, 29, 81, 114, 58, 6, 112], [7], [43, 95], [59], [24, 76], [39, 91], [41, 93], [103], [28, 80], [123], [118, 17, 69], [18, 70, 111], [131], [128], [124, 129], [130], [31, 83], [132], [126, 127, 133], [22, 74], [56, 108], [57, 109, 121, 55, 107, 116, 1, 4], [110, 119, 125, 117, 120], [52, 104, 32, 84, 16, 68], [54, 106], [62], [75], [82], [73], [30, 23, 87, 25, 35, 8, 9, 21, 10, 38], [61, 71, 72, 19, 20, 115], [40, 92]]

feature_map = [[32, 68, 16, 113, 84], [32, 34, 68, 16, 86], [64, 38, 10, 77, 25], [2, 5, 70, 111, 18], [6, 79, 114, 58, 27], [33, 70, 111, 85, 123], [1, 67, 4, 107, 15], [101, 81, 49, 88, 29], [4, 42, 108, 56, 94], [0, 3, 70, 111, 18], [71, 72, 73, 60, 61], [99, 47, 117, 120, 125], [37, 70, 111, 89, 123], [133, 103, 51, 126, 127], [66, 69, 14, 17, 118], [96, 133, 44, 126, 127], [133, 41, 63, 93, 127], [8, 9, 10, 11, 12], [98, 132, 133, 46, 126], [65, 133, 13, 126, 127], [38, 9, 10, 77, 25], [102, 105, 50, 53, 29], [122, 103, 78, 22, 26], [38, 8, 9, 10, 90], [128, 100, 133, 48, 127], [97, 1, 4, 45, 116], [122, 103, 74, 22, 26], [103, 74, 110, 22, 122], [36, 6, 112, 49, 81, 114, 88, 58, 29], [32, 7, 84, 122, 59], [133, 43, 95, 126, 127], [32, 68, 16, 84, 59], [1, 76, 109, 24, 57], [133, 39, 91, 126, 127], [133, 41, 93, 126, 127], [133, 103, 120, 126, 127], [133, 109, 80, 28, 127], [70, 111, 18, 116, 123], [130, 69, 17, 118, 119], [70, 107, 111, 18, 116], [131, 116, 117, 120, 125], [128, 133, 120, 126, 127], [129, 1, 4, 121, 124], [130, 117, 119, 120, 125], [109, 83, 117, 120, 31], [132, 133, 117, 120, 126], [133, 117, 120, 126, 127], [74, 110, 117, 22, 120], [1, 4, 108, 116, 56], [121, 1, 4, 107, 109, 116, 55, 57], [110, 117, 119, 120, 125], [32, 68, 104, 16, 52, 84], [106, 83, 54, 120, 31], [38, 8, 9, 10, 62], [38, 10, 75, 21, 23], [38, 9, 10, 82, 30], [71, 72, 73, 61, 62], [35, 38, 8, 9, 10, 23, 21, 87, 25, 30], [71, 72, 19, 20, 115, 61], [103, 40, 22, 122, 92]]
print(len(feature_map))
K = kit.KitNET(train.shape[1],maxAE,0,0, feature_map=feature_map)
print("Running KitNET:")
start = time.time()
# Here we process (train/execute) each individual observation.
# In this way, X is essentially a stream, and each observation is discarded after performing process() method.
for x in range(10):
    for i in range(train.shape[0]):
        loss = K.train(train[i,]) #will train during the grace periods, then execute on all the rest.
        
        if i % 1000 == 0:
            print(x, i, loss)

RMSEs = np.zeros(test.shape[0]) # a place to save the scores

for i in range(test.shape[0]):
    if i % 1000 == 0:
        print(i)
    RMSEs[i] = K.execute(test[i,])

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