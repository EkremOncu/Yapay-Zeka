
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('points.csv')
dataset = df.to_numpy(dtype='float32')


from sklearn.cluster import DBSCAN

dbs = DBSCAN(eps=2.5)
dbs.fit(dataset)

nclusters = np.max(dbs.labels_) + 1;

if nclusters == -1:
    nclusters = 0


plt.title('DBSCAN Clustered Points', fontsize=12)
for i in range(nclusters):
    plt.scatter(dataset[dbs.labels_ == i, 0], dataset[dbs.labels_ == i, 1])     

plt.scatter(dataset[dbs.labels_ == -1, 0], dataset[dbs.labels_ == -1, 1], marker='x', color='black')

legends = [f'Cluster-{i}' for i in range(1, nclusters + 1)]
legends.append('Noise Points')
plt.legend(legends)
plt.show()




