import pandas as pd

df = pd.read_csv('iris.csv')
dataset = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].to_numpy('float32')

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(dataset)
transformed_dataset = ss.transform(dataset)

from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score

ss_list = []
for i in range(2, 10):
    labels = KMeans(n_clusters=i, n_init=10).fit(dataset).labels_
    ss = silhouette_score(dataset, labels)
    ss_list.append(ss)
    print(f'{i} => {ss}')
    
optimal_cluster = np.argmax(ss_list) + 3
print(f'Optimal cluster: {optimal_cluster}' )

