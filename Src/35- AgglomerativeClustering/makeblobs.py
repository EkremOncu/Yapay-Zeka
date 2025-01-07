NCLUSTERS = 3

from sklearn.datasets import make_blobs

dataset, labels = make_blobs(100, 3, cluster_std=1, centers=NCLUSTERS)

from sklearn.cluster import KMeans

KMeans(NCLUSTERS, )

