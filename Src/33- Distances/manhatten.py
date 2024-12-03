import numpy as np

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

x = np.array([1, 1, 5, 6, 7, 8])
y = np.array([2, 2, 3, 6, 4, 6])

dist = manhattan_distance(x, y)
print(dist)

from scipy.spatial.distance import cityblock

dist = cityblock(x, y)
print(dist)
