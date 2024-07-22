from sklearn.preprocessing import OneHotEncoder
import numpy as np

a = np.array(['Mavi', 'Yeşil', 'Kırmızı', 'Mavi', 'Kırmızı', 'Mavi', 'Yeşil'])

ohe = OneHotEncoder()
result = ohe.fit_transform(a.reshape(-1, 1))

print(result)
print()
print(type(result))
print()
print(result.todense())