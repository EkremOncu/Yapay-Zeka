NNEIGHBORS = 5
THRESHOLD_STD = 2

import numpy as np
import pandas as pd

df = pd.read_csv('iris.csv')
dataset = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].to_numpy('float32')

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(dataset)
scaled_dataset = ss.transform(dataset)

from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=NNEIGHBORS)
nn.fit(dataset)
distances, indices = nn.kneighbors(dataset)

mean_distances = distances[:, 1:].mean(axis=1)

mean_mean_distances = np.mean(mean_distances)
std_mean_distances = np.std(mean_distances, ddof=0)
threshold = mean_mean_distances + THRESHOLD_STD * std_mean_distances

normal_points = dataset[mean_distances <= threshold]
anomaly_points = dataset[mean_distances > threshold]
normal_indices = np.where(mean_distances <= threshold)
anomaly_indices = np.where(mean_distances > threshold)

print(f'Anomaly points:\n{anomaly_points}')
print(f'Anomaly indices: {anomaly_indices}')

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_dataset = pca.fit_transform(dataset)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.title('Clustered Points')

plt.title('k-NN Anomaly Detection', fontsize=12)
plt.scatter(reduced_dataset[normal_indices, 0], reduced_dataset[normal_indices, 1])     

plt.scatter(reduced_dataset[anomaly_indices, 0], reduced_dataset[anomaly_indices, 1], marker='x', color='red')

plt.show()







