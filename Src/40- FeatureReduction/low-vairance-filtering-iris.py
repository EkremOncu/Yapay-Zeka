import pandas as pd

df = pd.read_csv('iris.csv')
dataset_x = df[['SepalLengthCm', 'PetalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
dataset_y = df['Species']

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
scaled_dataset_x = mms.fit_transform(dataset_x)

import numpy as np

feature_vars = np.var(scaled_dataset_x, axis=0)
sorted_arg_vars = np.argsort(feature_vars)
reduced_dataset_x = np.delete(scaled_dataset_x, sorted_arg_vars[:1], axis=1)
print(reduced_dataset_x)