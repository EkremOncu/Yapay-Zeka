NCLUSTERS = 5
BATCH_SIZE = 10


import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder


mbkm = MiniBatchKMeans(n_clusters=NCLUSTERS, batch_size=BATCH_SIZE)


numeric_features = ['Age', 'Income']
categoric_features = ['Education', 'Occupation', 'Settlement size']
binary_features = ['Sex', 'Marital status']


unique_list = [set(), set(), set()]
count = 0
total = 0
total_square = 0
for df in pd.read_csv('segmentation data.csv', chunksize=BATCH_SIZE):
    unique_list[0].update(df['Education'])
    unique_list[1].update(df['Occupation'])
    unique_list[2].update(df['Settlement size'])
    
    total += df[numeric_features].sum()
    total_square += (df[numeric_features] ** 2).sum()
    count += df.shape[0]
    
numeric_mean = total / count
numeric_std = (total_square / count - numeric_mean  ** 2) ** 0.5


unique_list = [list(us) for us in unique_list]
ohe = OneHotEncoder(sparse_output=False, categories=unique_list)


for df in pd.read_csv('segmentation data.csv', chunksize=BATCH_SIZE):
    categoric_array = ohe.fit_transform(df[categoric_features])
    numeric_array = ((df[numeric_features] - numeric_mean) / numeric_std).to_numpy()
    binary_array = df[binary_features].to_numpy()
    combined_array = np.hstack((categoric_array, binary_array, numeric_array))
    mbkm.partial_fit(combined_array)


print(mbkm.labels_)        


