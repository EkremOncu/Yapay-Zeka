import numpy as np
import pandas as pd

df = pd.read_csv('dataset.csv')
dataset = df.to_numpy('float32')

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
scaled_dataset = ss.fit_transform(dataset)

pca_dataset = scaled_dataset - np.mean(scaled_dataset, axis=0)

cmat = np.cov(pca_dataset, rowvar=False)

evals, evects = np.linalg.eig(cmat)

max_index = np.argmax(evals)
reduced_dataset = np.matmul(pca_dataset, evects[:, max_index].reshape((-1, 1)))
print(reduced_dataset)
