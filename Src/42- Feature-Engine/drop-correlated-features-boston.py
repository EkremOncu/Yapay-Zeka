CORR_THRESHOLD = 0.70

import pandas as pd

df = pd.read_csv('housing.csv', delimiter=r'\s+', header=None)

dataset_x = df.iloc[:, :-1].to_numpy(dtype='float32')
dataset_y = df.iloc[:, -1].to_numpy(dtype='float32')

from feature_engine.selection import DropCorrelatedFeatures

dcf = DropCorrelatedFeatures(method='pearson', threshold=CORR_THRESHOLD)

reduced_dataset_x = dcf.fit_transform(df)

print(reduced_dataset_x.shape)


"""
Özellik seçimi ve özwllik mühendisliği için kullanılan alternatif birtakım kütüphaneler 
de vardır.Özellik seçimi ve mühendisliği için son yıllarda gelime göstermiş olan 
kütüphanelerden biri de "feature-engine" isimli kütüphanedir.


feature-engine kütüphanesindeki DropCorrelatedFeatures sınıfı yüksek korelasyon 
filtrelemesi yapmaktadır. Biz bu sınıf türünden bir nesne yaratırken korelasyon 
için eşik değerini veririz. Korelasyon katsayısının nasıl hesaplanacağını
(tipik olarak "pearson") fonksiyonda belirtiriz.


Burada 0.70'lik bir eşik değeri seçildiğinde Boston Housing Prices veri kümesi 
10 sütuna indirgenmiştir. 
"""