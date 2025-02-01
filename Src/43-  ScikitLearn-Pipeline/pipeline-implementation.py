class SimplePipeline:
    def __init__(self, steps):
        self._steps = steps
        self.named_steps = dict(steps) 

    def fit(self, X, y=None):
        for name, step in self._steps:
            X = step.fit_transform(X, y)
        return self

    def transform(self, X):
        for name, step in self._steps:
            X = step.transform(X)
        return X

    def fit_transform(self, X, y=None):
        for name, step in self._steps:
            X = step.fit_transform(X, y)
        return X
    
    def __getitem__(self, index):
        return self._steps[index][1]

    def predict(self, X):
        # Son adımda tahmin yapılması varsayılır (örneğin, modelin son adımı)
        last_step = self._steps[-1][1]
        return last_step.predict(X)

NCLUSTERS = 3

import pandas as pd

df = pd.read_csv('iris.csv')
dataset = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].to_numpy('float32')

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


steps = [('Imputation', SimpleImputer(strategy='mean')), ('Scaling', StandardScaler()), ('Clustering', KMeans(3))]

pl = SimplePipeline(steps)
pl.fit(dataset)
distances = pl.transform(dataset)
print(distances)

km = pl.named_steps['Clustering']
print(km.labels_)

print(pl[2].labels_)
