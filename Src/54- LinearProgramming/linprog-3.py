import numpy as np
from scipy.optimize import linprog

c = np.array([-45, -55])
aub = np.array([[6, 4], [3, 10]])
bub = np.array([120, 180])

result = linprog(c, aub, bub)
if result.success:
    print(result.x)
    print(-result.fun)
else:
    print('Optimal çözüm yok')
print('-' * 30)
print(result)  
