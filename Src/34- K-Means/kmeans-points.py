NCLUSTERS = 3

import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv('points.csv')
dataset = df.to_numpy(dtype='float32')

km = KMeans(n_clusters=NCLUSTERS, n_init=10)
km.fit(dataset)

print(f'Inertia: {km.inertia_}')
print('-' * 20)

for i in range(NCLUSTERS):
    print(f'Cluster {i}', end='\n\n')
    cluster = dataset[km.labels_ == i]
    print(cluster)
    print('-' * 20)
    
print('Dataset', end='\n\n')
df['Cluster'] = km.labels_
print(df)

print('-' * 20)
print('Centroids')
print(km.cluster_centers_)



import matplotlib.pyplot as plt

plt.title('Clustered Points', fontsize=12)

for i in range(NCLUSTERS):
    plt.scatter(dataset[km.labels_ == i, 0], dataset[km.labels_ == i, 1])
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 60, color='red', marker='s')

plt.show()


# prediction

import numpy as np

predict_dataset = np.array([[3, 4], [4, 6], [6, 2]], dtype='float32')
predict_result = km.predict(predict_dataset)
print(f'Predict result: {predict_result}')

cluster_colors = np.array(['orange', 'green', 'magenta'])

plt.title('Clustered Points With Predicted Data', fontsize=12)

for i in range(NCLUSTERS):
    plt.scatter(dataset[km.labels_ == i, 0], dataset[km.labels_ == i, 1], color=cluster_colors[i])
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 60, color='red', marker='s')

plt.scatter(predict_dataset[:, 0], predict_dataset[:, 1], 60, color=cluster_colors[predict_result], marker='^')

plt.show()
                            













