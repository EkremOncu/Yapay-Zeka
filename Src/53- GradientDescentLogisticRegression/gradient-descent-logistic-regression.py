import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv('logistic-points.csv')

dataset_x = df.iloc[:, :-1].to_numpy()
dataset_y = df.iloc[:, -1].to_numpy()

ss = StandardScaler()
scaled_dataset_x = ss.fit_transform(dataset_x)

scaled_dataset_x = np.append(scaled_dataset_x, np.ones((dataset_x.shape[0], 1)), axis=1)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.title('Points for Logistic Regression')
plt.scatter(dataset_x[dataset_y == 0, 0], dataset_x[dataset_y == 0, 1], color='blue', marker='o')
plt.scatter(dataset_x[dataset_y == 1, 0], dataset_x[dataset_y == 1, 1], color='green', marker='^')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(['class 0', 'class 1'])

plt.show()

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def gradient_descent_logistic(dataset_x, dataset_y, learning_rate=0.001, niter=50000):
        b = np.zeros((dataset_x.shape[1], 1))
        for k in range(niter):
            h = sigmoid(dataset_x @ b)
            error = h - dataset_y
            grad = dataset_x.T @ error
            b = b - learning_rate * grad
        return b

b = gradient_descent_logistic(scaled_dataset_x, dataset_y.reshape(-1, 1))

x1 = np.linspace(-5, 5, 1000)
x2 = (-b[2] - b[0] * x1) / b[1]

points = np.vstack((x1, x2)).T
transformed_points = ss.inverse_transform(points)

plt.figure(figsize=(10, 8))
plt.title('Points for Logistic Regression')
plt.scatter(dataset_x[dataset_y == 0, 0], dataset_x[dataset_y == 0, 1], color='blue', marker='o')
plt.scatter(dataset_x[dataset_y == 1, 0], dataset_x[dataset_y == 1, 1], color='green', marker='^')
plt.plot(transformed_points[:, 0], transformed_points[:, 1], color='red')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(['class 0', 'class 1', 'regression line'])
plt.show()

