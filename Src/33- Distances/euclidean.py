import numpy as np

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

x = np.array([1, 1, 5, 6, 7, 8])
y = np.array([2, 2, 3, 6, 4, 6])

dist = euclidean_distance(x, y)
print(dist)

from scipy.spatial.distance import euclidean

dist = euclidean(x, y)
print(dist)
