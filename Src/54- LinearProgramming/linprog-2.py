import numpy as np
from scipy.optimize import linprog

c = np.array([-3, -2])
aub = np.array([[1, 2], [2, 1], [-1, 1], [0, 1]])
bub = np.array([6, 8, 1, 2])

result = linprog(c, aub, bub)
if result.success:
    print(result.x)
    print(-result.fun)
else:
    print('Optimal çözüm yok')
print('-' * 30)
print(result)  
