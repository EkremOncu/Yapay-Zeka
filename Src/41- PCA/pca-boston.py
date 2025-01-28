import pandas as pd

df = pd.read_csv('housing.csv', delimiter=r'\s+', header=None)

dataset_x = df.iloc[:, :-1].to_numpy(dtype='float32')
dataset_y = df.iloc[:, -1].to_numpy(dtype='float32')

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(dataset_x)
scaled_dataset_x = ss.transform(dataset_x)

from sklearn.decomposition import PCA

pca = PCA(10)

reduced_dataset_x = pca.fit_transform(scaled_dataset_x)
print(reduced_dataset_x)

print(dataset_x.shape)
print(reduced_dataset_x.shape)


"""
Yukarıdaki örnekte 13 sütundan oluşan "Boston Housing Prices" veri kümesi önce 
StandardScaler sınıfı ile ölçeklendirilmiştir. Daha sonra standardize edilmiş veri 
kümesi PCA işlemine sokularak 10 sütuna indirgenmiştir. Daha sonra da yapay sinir ağı 
yoluyla eğitim, test ve kestirim işlemleri yapılmıştır. Burada 13 sütunun 10 sütuna 
indirilmesinin önemli bir avantajı muhtemelen olmayacaktır.  

"""