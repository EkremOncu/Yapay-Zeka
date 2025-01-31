import pandas as pd

df = pd.read_csv('iris.csv')
dataset = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].to_numpy('float32')

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

pl = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), KMeans(3))
pl.fit(dataset)
distances = pl.transform(dataset)
print(distances)

km = pl.named_steps['kmeans']
print(km.labels_)

