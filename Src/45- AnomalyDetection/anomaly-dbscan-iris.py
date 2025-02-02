EPS = 0.70
MIN_SAMPLES = 5

import pandas as pd

df = pd.read_csv('iris.csv')
dataset = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].to_numpy('float32')

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(dataset)
transformed_dataset = ss.transform(dataset)

from sklearn.cluster import DBSCAN

dbs = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
dbs.fit(transformed_dataset)

import numpy as np

nclusters = np.max(dbs.labels_) + 1

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_dataset = pca.fit_transform(dataset)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.title('Clustered Points')

plt.title('DBSCAN Clustered Points', fontsize=12)
for i in range(nclusters):
    plt.scatter(reduced_dataset[dbs.labels_ == i, 0], reduced_dataset[dbs.labels_ == i, 1])     

plt.scatter(reduced_dataset[dbs.labels_ == -1, 0], reduced_dataset[dbs.labels_ == -1, 1], marker='x', color='black')

legends = [f'Cluster-{i}' for i in range(1, nclusters + 1)]
legends.append('Noise Points')
plt.legend(legends, loc='lower right')

plt.show()

anomaly_data = dataset[dbs.labels_ == -1]

print(f'Number of points with anomly: {len(anomaly_data)}')



