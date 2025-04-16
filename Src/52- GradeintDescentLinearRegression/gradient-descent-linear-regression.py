import numpy as np
import pandas as pd

df = pd.read_csv('points.csv')
dataset_x = df['x']
dataset_y = df['y']

def loss(y_hat, y):
    return np.sum((y_hat - y) **2)

def linear_regression_gradient(x, y, *, niter = None, epsilon=1e-10, learning_rate):
    b0 = 0
    b1 = 0
    
    prevloss = 0
    count = 0

    while True:
        y_hat = b0 + b1 * x     
        df_b0 = np.sum(y_hat - y)
        df_b1 = np.sum(x * (y_hat - y))
        
        b0 = b0 - df_b0 * learning_rate
        b1 = b1 - df_b1 * learning_rate
        
        if niter != None:
           if count >= niter:
              break
           count += 1      
        
        nextloss = loss(b0 + b1 * x, y)
        if np.abs(prevloss - nextloss) < epsilon:
            break
        prevloss = nextloss
        
    return b0, b1

b0, b1 = linear_regression_gradient(dataset_x, dataset_y,  niter=100, learning_rate=0.001)

x = np.linspace(1, 15, 100)
y = b0 + b1 * x

import matplotlib.pyplot as plt

plt.title('Linear Regression with Gradient Descent')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(dataset_x, dataset_y, color='blue')
plt.plot(x, y, color='red')
plt.show()

print(f'Slope = {b1}, Intercept={b0}')
