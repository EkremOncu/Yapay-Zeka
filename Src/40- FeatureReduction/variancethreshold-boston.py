import pandas as pd

df = pd.read_csv('housing.csv', delimiter=r'\s+', header=None)

dataset_x = df.iloc[:, :-1].to_numpy(dtype='float32')
dataset_y = df.iloc[:, -1].to_numpy(dtype='float32')

print(dataset_x.shape)

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
mms.fit(dataset_x)
scaled_dataset_x = mms.transform(dataset_x)

from sklearn.feature_selection import VarianceThreshold

vt = VarianceThreshold(0.04)

reduced_dataset_x = vt.fit_transform(scaled_dataset_x)
print(reduced_dataset_x.shape)
