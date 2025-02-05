REDUCTION_FEATURES = 10
ANOMALY_RATIO = 0.0001

import numpy as np
import pandas as pd

df = pd.read_csv('creditcard.csv', dtype='float32')

dataset = df.iloc[:, 1:-1].to_numpy()
dataset_y = df.iloc[:, -1].to_numpy()

from sklearn.decomposition import PCA

pca = PCA(n_components=REDUCTION_FEATURES)
reduced_dataset = pca.fit_transform(dataset)
inversed_dataset = pca.inverse_transform(reduced_dataset)

def anomaly_scores(original_data, manipulated_data):
    return np.sum((original_data - manipulated_data) ** 2, axis=1) 
    
scores = anomaly_scores(dataset, inversed_dataset)
q = np.quantile(scores, 1 - ANOMALY_RATIO)
anomalies = dataset[scores > q]
normals = dataset[scores <= q]

pca = PCA(n_components=2)

reduced_anomalies = dataset[scores > q]
reduced_normals = dataset[scores <= q]

import matplotlib.pyplot as plt

plt.title('Anomalies')
plt.legend(['Normal points', 'Anomalies'])
figure = plt.gcf()
figure.set_size_inches((10, 8))
plt.scatter(reduced_normals[:, 0], reduced_normals[:, 1], color='blue')
plt.scatter(reduced_anomalies[:, 0], reduced_anomalies[:, 1], marker='x', color='red')
plt.show()

print(f'Number of anomaly pointes: {len(anomalies)}')



