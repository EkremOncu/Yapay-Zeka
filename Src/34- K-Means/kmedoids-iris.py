NCLUSTERS = 3

import pandas as pd

df = pd.read_csv('iris.csv')
dataset = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].to_numpy('float32')

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(dataset)
transformed_dataset = ss.transform(dataset)

from sklearn_extra.cluster import KMedoids

km = KMedoids(n_clusters=NCLUSTERS)
km.fit(transformed_dataset)

df['Cluster'] = km.labels_

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(dataset)
reduced_dataset = pca.transform(dataset)
reduced_centroids = pca.transform(km.cluster_centers_)

plt.title('Clustered Points', fontsize=12)
for i in range(NCLUSTERS):
    plt.scatter(reduced_dataset[km.labels_ == i, 0], reduced_dataset[km.labels_ == i, 1])    
plt.scatter(reduced_dataset[km.medoid_indices_, 0], reduced_dataset[km.medoid_indices_, 1], color='red')    
plt.show()

import numpy as np

predict_data = np.array([[5.0,3.5,1.6,0.6], [4.8,3.0,1.4,0.3], [4.6,3.2,1.4,0.2]], dtype='float32')
transformed_predict_data = ss.transform(predict_data)

predict_result = km.predict(transformed_predict_data)
print(predict_result)







