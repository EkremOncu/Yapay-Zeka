import numpy as np
    
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 6, 8, 10, 12])
z = np.array([3, 5, 1, 6, 8])

m = np.vstack((x, y, z))

result = np.cov(m, ddof=0)
print(result)

m = np.cov((x, y, z), ddof=0)
print(m)


