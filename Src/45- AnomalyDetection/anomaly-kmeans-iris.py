NANOMALY_POINTS = 5

import numpy as np
import pandas as pd

df = pd.read_csv('iris.csv')
dataset = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].to_numpy('float32')

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(dataset)
scaled_dataset = ss.transform(dataset)

from sklearn.cluster import KMeans

km = KMeans(n_clusters=1, n_init=10)
distances = km.fit_transform(scaled_dataset)

arg_sorted_distances = np.argsort(distances[:, 0])

anomaly_points = dataset[arg_sorted_distances[-NANOMALY_POINTS:]]
print(anomaly_points)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(dataset)
reduced_dataset = pca.transform(dataset)
reduced_normal_points = reduced_dataset[arg_sorted_distances[:-NANOMALY_POINTS]]
reduced_anomaly_points = reduced_dataset[arg_sorted_distances[-NANOMALY_POINTS:]]


scaled_centroids = ss.inverse_transform(km.cluster_centers_)
reduced_centroids = pca.transform(scaled_centroids)

plt.figure(figsize=(10, 8))
plt.title('K-Means Anomaly Detection')
plt.scatter(reduced_normal_points[:, 0], reduced_normal_points[:, 1], color='blue')
plt.scatter(reduced_anomaly_points[:, 0], reduced_anomaly_points[:, 1], color='red')

plt.legend(['Normal Points', 'Anomaly Points'])
plt.show()






