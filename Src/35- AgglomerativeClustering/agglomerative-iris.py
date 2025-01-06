NCLUSTERS = 3

import pandas as pd

df = pd.read_csv('iris.csv')
dataset = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].to_numpy('float32')

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(dataset)
transformed_dataset = ss.transform(dataset)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(dataset)
reduced_dataset = pca.transform(dataset)

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=NCLUSTERS)
ac.fit(transformed_dataset)

import matplotlib.pyplot as plt

plt.title('Agglomerative Clustered Points', fontsize=12)
for i in range(NCLUSTERS):
    plt.scatter(reduced_dataset[ac.labels_ == i, 0], reduced_dataset[ac.labels_ == i, 1])     
plt.show()

from sklearn.cluster import KMeans

km = KMeans(n_clusters=NCLUSTERS, n_init=10)
km.fit(transformed_dataset)

transformed_centroids = ss.inverse_transform(km.cluster_centers_)
reduced_centroids = pca.transform(transformed_centroids)

plt.title('K-Means Clustered Points', fontsize=12)
for i in range(NCLUSTERS):
    plt.scatter(reduced_dataset[km.labels_ == i, 0], reduced_dataset[km.labels_ == i, 1])    
plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], color='red', marker='s')    
plt.show()

