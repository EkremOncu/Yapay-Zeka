NCLUSTERS = 3

import numpy as np
from scipy.spatial.distance import euclidean

def kmeans(dataset, nclusters, centroids=None):
    nrows = dataset.shape[0]
    clusters = np.full(nrows, -1)
    if centroids == None:
        centroids = rand_centroids(dataset, nclusters)

    change_flag = True
    while change_flag:
        change_flag = False
        for i in range(nrows):
            min_val = np.inf
            min_index = -1
            for k in range(nclusters):
                if not np.any(np.isnan(centroids[k])):
                    result = euclidean(dataset[i], centroids[k])
                    if result < min_val:
                        min_val = result
                        min_index = k
            if clusters[i] != min_index:
                change_flag = True
            clusters[i] = min_index

        for i in range(nclusters):
            idataset = dataset[clusters == i]
            centroids[i] = np.mean(idataset, axis=0) if len(idataset) else np.nan

    dataset_clusters = []
    inertia = 0
    for i in range(nclusters):
        idataset = dataset[clusters == i]
        dataset_clusters.append(idataset)
        inertia += np.sum((idataset - centroids[i]) ** 2) if len(idataset) > 0 else 0

    return clusters, dataset_clusters, centroids, inertia

def rand_centroids(dataset, nclusters):
    ncols = dataset.shape[1]
    centroids = np.zeros((nclusters, ncols), dtype=np.float32)
    for i in range(ncols):
        maxi = np.max(dataset[:, i])
        mini = np.min(dataset[:, i])
        rangei = maxi - mini
        centroids[:, i] = mini + rangei * np.random.random(nclusters)

    return centroids

import pandas as pd

df = pd.read_csv('points.csv', dtype='float32')
dataset = df.to_numpy(dtype='float32')

clusters, dataset_clusters, centroids, inertia = kmeans(dataset, NCLUSTERS)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.title('Clustered Points')

for i in range(NCLUSTERS):
    plt.scatter(dataset_clusters[i][:, 0], dataset_clusters[i][:, 1], color='ygb'[i])    

plt.scatter(centroids[:, 0], centroids[:, 1], 60, color='red', marker='s')
legends = [f'Cluster-{i}' for i in range(1, NCLUSTERS + 1)]
legends.append('Centroids')
plt.legend(legends)

plt.show()