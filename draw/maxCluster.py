import matplotlib.pyplot as plt

# Data
maxClusters = [1, 4, 7, 10, 13, 16, 19, 21, 24]
f1_scores = [0.8642, 0.8906, 0.8888, 0.8927, 0.8623, 0.8564, 0.8557, 0.8430, 0.8407]
train_time = [8518.28, 7665.71, 7111.09, 7375.26, 7327.92, 6858.76, 6829.17, 6610.69, 6571.61]
test_time = [9578.24, 7650.86, 6680.92, 7116.96, 6601.58, 6040.94, 5855.58, 5591.3, 5325.15]

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot F1 Scores on the primary y-axis
ax1.plot(maxClusters, f1_scores, label='F1 Score', color='blue', marker='o')
ax1.set_xlabel('Max Clusters')
ax1.set_ylabel('F1 Score', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(0.825, 0.925)  # Adjusting the F1 score axis to range from 0.825 to 0.925

# Create a secondary y-axis for time metrics
ax2 = ax1.twinx()
ax2.plot(maxClusters, train_time, label='Training Time (ms)', color='green', linestyle='--', marker='o')
ax2.plot(maxClusters, test_time, label='Testing Time (ms)', color='red', linestyle='--', marker='o')
ax2.set_ylabel('Time (ms)', color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(4000, 10000)  # Adjusting the time axis to range from 4000 to 10000

# Adding title and legends
fig.suptitle('Model Performance and Runtime Metrics')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.grid(True)

# Show plot
plt.savefig("maxCluster.png")