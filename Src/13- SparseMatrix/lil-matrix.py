import numpy as np
from scipy.sparse import lil_matrix

a = np.array([[0, 0, 10, 20, 0], [15, 0, 0, 0, 40], [12, 0, 51, 0, 16], [42, 0, 18, 0, 16], [0, 0, 0, 0, 0]])
print(a)
lil = lil_matrix(a, dtype='int32')

print("-----------------------")

print(lil)

