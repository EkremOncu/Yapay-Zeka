NCLUSTERS = 3

import pandas as pd

df = pd.read_csv('iris.csv')
dataset = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].to_numpy('float32')

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

steps = [('Imputation', SimpleImputer(strategy='mean')), ('Scaling', StandardScaler()), ('Clustering', KMeans(3))]

pl = Pipeline(steps)
distances = pl.fit_transform(dataset)
print(distances)

km = pl.named_steps['Clustering']
print(km.labels_)

print(pl[2].labels_)
