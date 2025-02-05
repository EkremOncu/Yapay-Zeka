from sklearn.datasets import make_blobs

dataset, clusters = make_blobs(n_samples=100, cluster_std=2, centers=1)

import matplotlib.pyplot as plt

plt.title('Random Points')
figure = plt.gcf()
figure.set_size_inches((10, 8))
plt.scatter(dataset[:, 0], dataset[:, 1], color='blue')
plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
reduced_dataset = pca.fit_transform(dataset)
inversed_dataset = pca.inverse_transform(reduced_dataset)

def anomaly_scores(original_data, manipulated_data):
    loss = np.sum((original_data - manipulated_data) ** 2, axis=1) 
    loss = (loss  - np.min(loss)) / (np.max(loss) - np.min(loss))

    return loss

scores = anomaly_scores(dataset, inversed_dataset)

import numpy as np

ANOMALY_RATIO = 0.05

q = np.quantile(scores, 1 - ANOMALY_RATIO)
anomalies = dataset[scores > q]

import matplotlib.pyplot as plt

plt.title('Anomalies')
plt.legend(['Normal points', 'Anomalies'])
figure = plt.gcf()
figure.set_size_inches((10, 8))
plt.scatter(dataset[:, 0], dataset[:, 1], color='blue')
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='red')
plt.show()
