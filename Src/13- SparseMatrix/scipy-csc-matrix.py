import numpy as np
from scipy.sparse import csc_matrix

a = np.array([[0, 0, 10, 20, 0], [15, 0, 0, 0, 40], [12, 0, 51, 0, 16], [42, 0, 18, 0, 16], [0, 0, 0, 0, 0]])
print(a)
print("-"*20)

csc = csc_matrix(a, dtype='int32')
print(csc)
print("-"*20)

print(csc.todense())
print("-"*20)


print(f'data: {csc.data}')
print(f'indices: {csc.indices}')
print(f'indices: {csc.indptr}')

