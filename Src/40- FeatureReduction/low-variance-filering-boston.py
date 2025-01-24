import pandas as pd

df = pd.read_csv('housing.csv', delimiter=r'\s+', header=None)

dataset_x = df.iloc[:, :-1].to_numpy(dtype='float32')
dataset_y = df.iloc[:, -1].to_numpy(dtype='float32')

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
mms.fit(dataset_x)
scaled_dataset_x = mms.transform(dataset_x)

import numpy as np

feature_vars = np.var(scaled_dataset_x, axis=0)
sorted_arg_vars = np.argsort(feature_vars)
reduced_dataset_x = np.delete(dataset_x, sorted_arg_vars[:5], axis=1)
print(reduced_dataset_x)
