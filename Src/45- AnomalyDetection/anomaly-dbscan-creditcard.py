EPS = 10

import pandas as pd

df = pd.read_csv('creditcard.csv', dtype='float32')

dataset = df.iloc[:100000, 1:-1].to_numpy()
dataset_y = df.iloc[:100000, -1].to_numpy()

import numpy as np
from sklearn.cluster import DBSCAN

dbs = DBSCAN(eps=EPS, min_samples=5)
dbs.fit(dataset)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_dataset = pca.fit_transform(dataset)

nclusters = np.max(dbs.labels_) + 1


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.title('Clustered Points')
for i in range(nclusters):
    plt.scatter(reduced_dataset[dbs.labels_ == i, 0], reduced_dataset[dbs.labels_ == i, 1])    

plt.scatter(reduced_dataset[dbs.labels_ == -1, 0], reduced_dataset[dbs.labels_ == -1, 1], marker='x', color='black')

legends = [f'Cluster-{i}' for i in range(1, nclusters + 1)]
legends.append('Noise Points')
plt.legend(legends, loc='lower right')
plt.show()

anomaly_data = dataset[dbs.labels_ == -1]
print(f'Number of points with anomly: {len(anomaly_data)}')

original_anomaly_indices = np.where(dataset_y == 1)
dbscan_anomaly_indices = np.where(dbs.labels_ == -1)
intersect_anomalies = np.intersect1d(original_anomaly_indices, dbscan_anomaly_indices)

success_ratio = len(intersect_anomalies) / (dataset_y == 1).sum()
print(f'Success ratio: {success_ratio}')        # %26

