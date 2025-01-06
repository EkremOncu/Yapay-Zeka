NCLUSTERS = 3

import pandas as pd

df = pd.read_csv('points.csv')

dataset = df.to_numpy(dtype='float32')

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=NCLUSTERS, linkage='ward', compute_distances=True)

result = ac.fit_predict(dataset)

print(result)
