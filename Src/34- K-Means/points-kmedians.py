NCLUSTERS = 3

import pandas as pd
from pyclustering.cluster.kmedians import kmedians

df = pd.read_csv('points.csv')
dataset = df.to_numpy(dtype='float32')

km = kmedians(dataset, initial_medians=[[5, 4], [1, 2]])
km.process()

clusters = km.get_clusters()
print(clusters)
