NCLUSTERS = 3

from sklearn.datasets import make_blobs

dataset, _ = make_blobs(100, 3, cluster_std=1, centers=NCLUSTERS)

from sklearn.cluster import KMeans, AgglomerativeClustering

km = KMeans(NCLUSTERS, n_init=10)
km.fit(dataset)

ac = AgglomerativeClustering(NCLUSTERS)
ac.fit(dataset)

import matplotlib.pyplot as plt

plt.title('Agglomerative Clustering')
for i in range(NCLUSTERS):
    plt.scatter(dataset[ac.labels_ == i, 0], dataset[ac.labels_ == i, 1])
plt.show()

plt.title('K-Means Clustering')
for i in range(NCLUSTERS):
    plt.scatter(dataset[km.labels_ == i, 0], dataset[km.labels_ == i, 1])
plt.show()



