import numpy as np

def cov(x, y):
    return np.sum((x - np.mean(x)) * (y - np.mean(y))) / len(x)
            
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 8, 9, 12])

result = cov(x, y)
print(result)
    
result = np.cov(x, y, ddof=0)
print(result)

result = np.corrcoef((x, y))
print(result)

import matplotlib.pyplot as plt

plt.title('Covariance')
plt.scatter(x, y)
plt.show()
