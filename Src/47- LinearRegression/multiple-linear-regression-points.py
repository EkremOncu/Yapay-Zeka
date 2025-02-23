import numpy as np
import pandas as pd

def multiple_linear_regression(x, y):
    ones = np.ones((x.shape[0], 1))
    x = np.concatenate((ones, x), axis=1)
    betas = (np.linalg.inv(x.T @ x) @ x.T @ y).reshape(-1)
    return betas[0], betas[1:]
    
df = pd.read_csv('points.csv')
dataset_x = df.iloc[:, :-1].to_numpy()
dataset_y = df.iloc[:, -1].to_numpy().reshape(-1, 1)

intercept, coeffs = multiple_linear_regression(dataset_x, dataset_y)

x = np.linspace(0, 15, 1000)
y = intercept + coeffs[0] * x

import matplotlib.pyplot as plt

plt.title('Simple Linear Regression')
plt.scatter(dataset_x, dataset_y, color='blue')
plt.plot(x, y, color='red')
plt.xticks(range(1, 15))
plt.show()

predict_data = np.array([3])
result = np.dot(coeffs, predict_data) + intercept
print(result)

    