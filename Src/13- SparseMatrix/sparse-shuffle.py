import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle

a = np.array([[0, 0, 10, 20, 0], [15, 0, 0, 0, 40], [12, 0, 51, 0, 16], [42, 0, 18, 0, 16], [0, 0, 0, 0, 0]])

csr = csr_matrix(a)
y = [10, 20, 30, 40, 50]
result_x, result_y = shuffle(csr, y)

print(csr.todense())
print('-' * 15)


print(result_x.todense())
print('-' * 15)

print(result_y)


"""
shuffle fonksiyonu birden fazla girdi alabilmektedir. Karıştırmayı paralel biçimde 
yaptığı için karıştırılmış x matrisiyle y matrisinin karşılıklı elemanları yine 
aynı olmaktadır.

"""