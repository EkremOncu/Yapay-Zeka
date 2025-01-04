NCLUSTERS = 3

import pandas as pd

df = pd.read_csv('points.csv')

dataset = df.to_numpy(dtype='float32')

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=NCLUSTERS, linkage='ward', compute_distances=True)

ac.fit(dataset)

df['Cluster'] = ac.labels_

import matplotlib.pyplot as plt

for i in range(NCLUSTERS):
    plt.scatter(dataset[ac.labels_ == i, 0], dataset[ac.labels_ == i, 1])
    
plt.show()

