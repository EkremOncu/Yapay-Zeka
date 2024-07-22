import numpy as np
from scipy.sparse import csc_matrix

a = np.array([[0, 0, 10, 20, 0], [15, 0, 0, 0, 40], [12, 0, 51, 0, 16], [42, 0, 18, 0, 16], [0, 0, 0, 0, 0]])

csc = csc_matrix(a, dtype='int32')

print(csc)
print('csc' * 15)

csr = csc.tocsr()
print(csr)
print('csr' * 15)


dok = csr.todok()
print(dok)
print('dok' * 15)


lil = dok.tolil()
print(lil)
print('lil' * 15)