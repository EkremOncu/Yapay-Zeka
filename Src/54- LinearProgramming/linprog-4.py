import numpy as np
from scipy.optimize import linprog

c = np.array([-2.25, 26, 0.21])
aub = np.array([[-1, -1, -10], [0, -10, -10], [-100, -100, -10], [70, 50, 120]])
bub = np.array([-15, -30, -10, 80]) 

result = linprog(c, aub, bub)
if result.success:
    print(result.x)
    print(-result.fun)
else:
    print('Optimal çözüm yok')
print('-' * 30)
print(result)  

