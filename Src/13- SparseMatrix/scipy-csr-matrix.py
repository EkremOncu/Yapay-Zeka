import numpy as np
from scipy.sparse import csr_matrix

a = np.array([[0, 0, 10, 20, 0], [15, 0, 0, 0, 40], [12, 0, 51, 0, 16], [42, 0, 18, 0, 16], [0, 0, 0, 0, 0]])
print(a)
print("-"*20)

csr = csr_matrix(a, dtype='int32')
print(csr)
print("-"*20)

print(csr.todense())
print("-"*20)


print(f'data: {csr.data}')
print(f'indices: {csr.indices}')
print(f'indices: {csr.indptr}')


