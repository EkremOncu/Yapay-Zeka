import pandas as pd

df = pd.read_csv('housing.csv', delimiter=r'\s+', header=None)

dataset_x = df.iloc[:, :-1].to_numpy(dtype='float32')
dataset_y = df.iloc[:, -1].to_numpy(dtype='float32')

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(dataset_x)
scaled_dataset_x = ss.transform(dataset_x)

import numpy as np
from sklearn.decomposition import PCA

total_ratios = []
for i in range(1, dataset_x.shape[1] + 1):
    pca = PCA(i)
    pca.fit(scaled_dataset_x)
    total_ratio = np.sum(pca.explained_variance_ratio_)
    total_ratios.append(total_ratio)
    
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.title('Optimal number of Featured')
plt.plot(range(1, dataset_x.shape[1] + 1), total_ratios, color='red')
plt.plot(range(1, dataset_x.shape[1] + 1), total_ratios, 'bo', color='blue')
plt.legend(['Total explained variance ratio'])
plt.xlabel('Nuber of Features')
plt.ylabel('Ratio')
plt.xticks(range(1, dataset_x.shape[1] + 1))

plt.show()
