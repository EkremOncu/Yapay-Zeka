import pandas as pd

df = pd.read_csv('iris.csv')
dataset = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].to_numpy('float32')

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class MyStandardScaler:
    def __init__(self):
        self._ss = StandardScaler()
        
    def fit(self, X, y=None):
        return self._ss.fit(X, y) 
    
    def transform(self, X):
        return self._ss.transform(X) 
    
    def fit_transform(self, X, y=None):
        return self._ss.fit_transform(X, y) 

pl = make_pipeline(SimpleImputer(strategy='mean'), MyStandardScaler(), KMeans(3))
distances = pl.fit_transform(dataset)
print(distances)

km = pl.named_steps['kmeans']
print(km.labels_)