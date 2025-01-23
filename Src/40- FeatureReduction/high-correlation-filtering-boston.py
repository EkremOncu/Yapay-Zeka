CORR_THRESHOLD = 0.75

import pandas as pd

df = pd.read_csv('housing.csv', delimiter=r'\s+', header=None)

dataset_x = df.iloc[:, :-1].to_numpy(dtype='float32')
dataset_y = df.iloc[:, -1].to_numpy(dtype='float32')

import numpy as np

feature_corrs = np.abs(np.corrcoef(dataset_x, rowvar=False))

eliminated_features = []

for i in range(feature_corrs.shape[0]):
    for k in range(i):
        if i != k and i not in eliminated_features and feature_corrs[i, k] > CORR_THRESHOLD:
            eliminated_features.append(k)
            # print((i, k), feature_corrs[i, k])

print(eliminated_features)

reduced_dataset_x = np.delete(dataset_x, eliminated_features, axis=1)
print(reduced_dataset_x.shape)

