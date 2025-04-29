import numpy as np
from scipy.optimize import linprog

c = np.array([4, 6, 5, 2, 3, 4])
aub = np.array([
    [1, 1, 0, 0, 0, 0], 
    [0, 0, 1, 1, 0, 0], 
    [0, 0, 0, 0, 1, 1]])
bub = np.array([100, 150, 200])
aeq = np.array([[1, 0, 1, 0, 1, 0], 
                [0, 1, 0, 1, 0, 1]])
beq = np.array([180, 270])

result = linprog(c, aub, bub, aeq, beq)

if result.success:
    print(result.x)
    print(result.fun)
else:
    print('Optimal çözüm yok')
print('-' * 30)
print(result)  

