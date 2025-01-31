from sklearn.pipeline import Pipeline

def mymake_pipeline(*steps, memory=None, verbose=False):
    corrected_steps = [(type(step).__name__.lower(), step) for step in steps] 
    return Pipeline(corrected_steps, memory=memory, verbose=verbose) 
    
NCLUSTERS = 3

import pandas as pd

df = pd.read_csv('iris.csv')
dataset = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].to_numpy('float32')

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

pl = mymake_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), KMeans(3))
pl.fit(dataset)
distances = pl.transform(dataset)
print(distances)

km = pl.named_steps['kmeans']
print(km.labels_)