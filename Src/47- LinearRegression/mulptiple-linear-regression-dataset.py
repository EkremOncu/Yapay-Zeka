import numpy as np
import pandas as pd

def multiple_linear_regression(x, y):
    ones = np.ones((x.shape[0], 1))
    x = np.concatenate((ones, x), axis=1)
    betas = (np.linalg.inv(x.T @ x) @ x.T @ y).reshape(-1)
    return betas[0], betas[1:]
    
df = pd.read_csv('data.csv')
dataset_x = df.iloc[:, :-1].to_numpy()
dataset_y = df.iloc[:, -1].to_numpy().reshape(-1, 1)

intercept, coeffs = multiple_linear_regression(dataset_x, dataset_y)

predict_data = np.array([3, 3, -17])

"""
total = intercept
for i in range(len(coeffs)):
    total += coeffs[i] * predict_data[i]
print(total)
"""

result = np.dot(coeffs, predict_data) + intercept
print(result)

    