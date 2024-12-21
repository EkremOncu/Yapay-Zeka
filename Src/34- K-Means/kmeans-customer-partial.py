BATCH_SIZE = 10

import pandas as pd

tfr = pd.read_csv('segmentation data.csv', chunksize=BATCH_SIZE)

for df in tfr:
    df.drop(labels=['ID'], axis=1, inplace=True)
    ohe_df = pd.get_dummies(df, columns=['Education', 'Occupation', 'Settlement size'], dtype='uint8')
    print(df)
    


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
scaled_dataset = ss.fit_transform(ohe_df.to_numpy())

from sklearn.cluster import KMeans

total_inertias = [KMeans(n_clusters=i, n_init=10).fit(scaled_dataset).inertia_ for i in range(1, 100)]


import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))
plt.title('Elbow Graph')
plt.plot(range(1, 100), total_inertias, marker='o')
plt.xticks(range(1, 100, 5))
plt.show()


import numpy as np
from sklearn.metrics import silhouette_score

optimal_cluster = np.argmax([silhouette_score(scaled_dataset, KMeans(i, n_init=10).fit(scaled_dataset).labels_) 
                             for i in range(2, 10)]) + 3

km = KMeans(n_clusters=optimal_cluster, n_init=100)
km.fit(scaled_dataset)


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
transformed_dataset = pca.fit_transform(scaled_dataset)

plt.figure(figsize=(12, 10))
plt.title('Clustered Points')
for i in range(1, optimal_cluster + 1):
    plt.scatter(transformed_dataset[km.labels_ == i, 0], transformed_dataset[km.labels_ == i, 1])
plt.show()









