import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv('points.csv')
dataset = df.to_numpy(dtype='float32')

inertias = [KMeans(n_clusters=i, n_init=10).fit(dataset).inertia_ for i in range(1, 10)]

import matplotlib.pyplot as plt

plt.title('Elbow Point Method', fontsize=12)
plt.plot(range(1, 10), inertias)
plt.show()


# Dirsek noktasının 3 olduğu tespit edilmiştir

km = KMeans(n_clusters=3, n_init=10)
km.fit(dataset)

plt.title('Clustered Points', fontsize=12)

for i in range(3):
    plt.scatter(dataset[km.labels_ == i, 0], dataset[km.labels_ == i, 1])
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 60, color='red', marker='s')

plt.show()