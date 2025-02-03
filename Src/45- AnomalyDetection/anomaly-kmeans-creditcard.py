NANOMALY_POINTS = 2000

import pandas as pd

df = pd.read_csv('creditcard.csv', dtype='float32')

dataset = df.iloc[:, 1:-1].to_numpy()
dataset_y = df['Class'].to_numpy()

import numpy as np
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=1)
distances = kmeans.fit_transform(dataset)

arg_sorted_distances = np.argsort(distances[:, 0])

anomaly_points = dataset[arg_sorted_distances[-NANOMALY_POINTS:]]
print(anomaly_points)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
decomposed_dataset = pca.fit_transform(dataset)

decomposed_normal_points = decomposed_dataset[arg_sorted_distances[:-NANOMALY_POINTS]]
decomposed_anomaly_points = decomposed_dataset[arg_sorted_distances[-NANOMALY_POINTS:]]

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.title('K-Means Anomaly Detection')
plt.scatter(decomposed_normal_points[:, 0], decomposed_normal_points[:, 1], color='blue')
plt.scatter(decomposed_anomaly_points[:, 0], decomposed_anomaly_points[:, 1], color='red')
plt.legend(['Normal Points', 'Anomaly Points'])
plt.show()

original_anomaly_indices = np.where(dataset_y == 1)
kmeans_anomaly_indices = arg_sorted_distances[-NANOMALY_POINTS:]

intersect_anomalies = np.intersect1d(original_anomaly_indices, kmeans_anomaly_indices)

print(len(intersect_anomalies))
