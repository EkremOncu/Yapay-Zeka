NCLUSTERS = 2

from sklearn.datasets import make_circles

dataset, labels = make_circles(100, factor=0.8, noise=0.05)

import matplotlib.pyplot as plt

plt.title('Points')
for i in range(2):
    plt.scatter(dataset[labels == i, 0], dataset[labels == i, 1])    
plt.show()



from sklearn.cluster import KMeans

km = KMeans(NCLUSTERS, n_init=10)
km.fit(dataset)

plt.title('K-Means Clustering')
for i in range(NCLUSTERS):
    plt.scatter(dataset[km.labels_ == i, 0], dataset[km.labels_ == i, 1])
plt.show()



from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(NCLUSTERS)
ac.fit(dataset)

import matplotlib.pyplot as plt

plt.title('Agglomerative Clustering')
for i in range(NCLUSTERS):
    plt.scatter(dataset[ac.labels_ == i, 0], dataset[ac.labels_ == i, 1])
plt.show()

import numpy as np

kmeans_ratio = np.sum(km.labels_ == labels) / len(labels)
agglomerative_ratio= np.sum(ac.labels_ == labels) / len(labels)

print(f'K-Means ratio: {kmeans_ratio}')
print(f'Agglomerative ratio: {agglomerative_ratio}')




from scipy.cluster.hierarchy import linkage, dendrogram

linkage_data= linkage(dataset)

plt.title('Points Dendrogram')
dendrogram(linkage_data)
plt.show()










