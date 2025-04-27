import numpy as np
from scipy.optimize import linprog

c = np.array([5, 3, 4])
aub = np.array([[-10, -4, -2], [-2, -5, -3], [-8, -4, -10 ]])
bub = np.array([-60, -40, -80])

result = linprog(c, aub, bub)
if result.success:
    print(result.x)
    print(result.fun)
else:
    print('Optimal çözüm yok')
print('-' * 30)
print(result)  
