NCLUSTERS = 3

import pandas as pd

df = pd.read_csv('iris.csv')
dataset = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].to_numpy('float32')

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(dataset)
transformed_dataset = ss.transform(dataset)


from sklearn.cluster import KMeans

inertias = [KMeans(n_clusters=i, n_init=10).fit(transformed_dataset).inertia_ for i in range(1, 10)]

import matplotlib.pyplot as plt

plt.title('Elbow Point Method', fontsize=12)
plt.plot(range(1, 10), inertias)
plt.show()