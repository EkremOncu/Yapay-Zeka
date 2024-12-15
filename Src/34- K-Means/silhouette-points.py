import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv('points.csv')
dataset = df.to_numpy(dtype='float32')

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

optimal_cluster = np.argmax([silhouette_score(dataset, KMeans(i, n_init=10).fit(dataset).labels_) 
                             for i in range(2, 10)]) + 3

print(f'Optimal cluster: {optimal_cluster}' )
    
"""
2 => 0.5544097423553467
3 => 0.47607627511024475
4 => 0.4601795971393585
5 => 0.4254012405872345
6 => 0.3836685121059418
7 => 0.29372671246528625
8 => 0.21625620126724243
9 => 0.11089805513620377

"""



