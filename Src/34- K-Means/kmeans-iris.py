NCLUSTERS = 3

import pandas as pd

df = pd.read_csv('iris.csv')
dataset = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].to_numpy('float32')

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(dataset)
transformed_dataset = ss.transform(dataset)


from sklearn.cluster import KMeans

km = KMeans(n_clusters=NCLUSTERS, n_init=10)
km.fit(transformed_dataset)

df['Cluster'] = km.labels_



import matplotlib.pyplot as plt

##############################################
"""
Şimdi kümelenmiş noktaların grafiğini çizdirelim. Ancak noktalar dört boyutlu uzaya 
ilişkin olduğu için onu iki boyuta indirgeyelim. Bu işleme "boyutsal özellik indirgemesi 
(dimentionality feature reduction)" denilmektedir. Bu konu ilerde ele alınacaktır. 
Bu işlem "temel bileşenler analizi (principle component analysis)" denilen yöntemle 
aşağıdaki gibi yapılabilir

"""

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(dataset)
reduced_dataset = pca.transform(dataset)

# çizdirirken orjinal dataseti kullanıyoruz, scale edilmemiş hali, onun için centroiti de öyle kullanmalıyız

transformed_centroids = ss.inverse_transform(km.cluster_centers_)
reduced_centroids = pca.transform(transformed_centroids)

# Artık iki boyuta indirgenmiş olan kümeleri saçılma diyagramıyla gösterebiliriz:

plt.title('Clustered Points', fontsize=12)
for i in range(NCLUSTERS):
    plt.scatter(reduced_dataset[km.labels_ == i, 0], reduced_dataset[km.labels_ == i, 1])    

plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], color='red', marker='s')    
plt.show()

##############################################



import numpy as np

predict_data = np.array([[5.0,3.5,1.6,0.6], [4.8,3.0,1.4,0.3], [4.6,3.2,1.4,0.2]], dtype='float32')
transformed_predict_data = ss.transform(predict_data)

predict_result = km.predict(transformed_predict_data)
print(predict_result)







