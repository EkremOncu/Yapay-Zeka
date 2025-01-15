import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('points.csv')
dataset = df.to_numpy(dtype='float32')

from sklearn.cluster import OPTICS

opt = OPTICS(min_samples=3, eps=1, cluster_method='xi')

opt.fit(dataset)
print(opt.labels_)
print(opt.reachability_)
print(opt.ordering_)
print(opt.core_distances_)
print(opt.cluster_hierarchy_)

nclusters = np.max(opt.labels_) + 1;

if nclusters == -1:
    nclusters = 0


plt.title('OPTICS Clustered Points', fontsize=12)

for i in range(nclusters):
    plt.scatter(dataset[opt.labels_ == i, 0], dataset[opt.labels_ == i, 1])     

plt.scatter(dataset[opt.labels_ == -1, 0], dataset[opt.labels_ == -1, 1], marker='x',
            color='black')

legends = [f'Cluster-{i}' for i in range(1, nclusters + 1)]
legends.append('Noise Points')
plt.legend(legends)
plt.show()

