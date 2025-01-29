import numpy as np
import pandas as pd

df = pd.read_csv('dataset.csv')
dataset = df.to_numpy('float32')

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
scaled_dataset = ss.fit_transform(dataset)

from sklearn.decomposition import PCA

pca = PCA(1)

reduced_dataset = pca.fit_transform(scaled_dataset)
print(reduced_dataset)
