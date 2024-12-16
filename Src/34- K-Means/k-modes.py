import pandas as pd

dataset_dict = {
    'Renk': ['Kırmızı', 'Mavi', 'Yeşil', 'Mavi', 'Kırmızı', 'Yeşil', 'Kırmızı', 'Mavi', 'Yeşil', 'Mavi'],
    'Cinsiyet': ['Kadın', 'Erkek', 'Kadın', 'Kadın', 'Erkek', 'Erkek', 'Kadın', 'Erkek', 'Kadın', 'Kadın'],
    'Ülke': ['Türkiye', 'Almanya', 'Fransa', 'İngiltere', 'Türkiye', 'Almanya', 'Fransa', 'Türkiye', 
             'İngiltere', 'Almanya']
}

dataset_df = pd.DataFrame(dataset_dict)

import numpy as np
from sklearn.metrics import silhouette_score
from kmodes.kmodes import KModes
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

encoded_df = pd.DataFrame()

for column in dataset_df.columns:
    encoded_df[column] = le.fit_transform(dataset_df[column])

optimal_cluster = np.argmax([silhouette_score(encoded_df, KModes(i, n_init=10).fit(dataset_df).labels_, 
                                              metric='hamming') for i in range(2, 8)]) + 3

km = KModes(n_clusters=optimal_cluster, n_init=10)
km.fit(dataset_df)

for i in range(optimal_cluster):
    cluster = dataset_df.iloc[km.labels_ == i]
    print(f'cluster {i}')
    print(cluster)
    print('-' * 10)






