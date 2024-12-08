import numpy as np
from scipy.spatial.distance import cosine

a = np.array([1, 0, 0, 1])
b = np.array([1, 1, 0, 0])

hdist = cosine(a, b)
print(hdist)                # 0.5