from sklearn.datasets import make_circles

dataset, labels = make_circles(100, factor=0.2, noise=0.02)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.title('Random Points')

for i in range(2):
    plt.scatter(dataset[labels == i, 0], dataset[labels == i, 1])    
plt.show()

import numpy as np
from sklearn.cluster import OPTICS

optics = OPTICS(min_samples=10)
optics.fit(dataset)

nclusters = np.max(optics.labels_) + 1


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.title('Clustered Points')

for i in range(nclusters):
    plt.scatter(dataset[optics.labels_ == i, 0], dataset[optics.labels_ == i, 1])

plt.scatter(dataset[optics.labels_ == -1, 0], dataset[optics.labels_ == -1, 1], 
            marker='x', color='black')

legends = [f'Cluster-{i}' for i in range(1, nclusters + 1)]
legends.append('Noise Points')
plt.legend(legends)

plt.show()