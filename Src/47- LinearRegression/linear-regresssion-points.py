import numpy as np
import pandas as pd

df = pd.read_csv('points.csv')

dataset_x = df['x'].to_numpy()
dataset_y = df['y'].to_numpy()

def linear_regression(x, y):
    a = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    b = np.sum(((x - np.mean(x)) ** 2))
    b1 = a / b
    b0 = (np.sum(y) - b1 * np.sum(x)) / len(x)
    return b0, b1

b0, b1 = linear_regression(dataset_x, dataset_y)

x = np.linspace(0, 15, 1000)
y = b0 + b1 * x

import matplotlib.pyplot as plt

plt.title('Simple Linear Regression')
plt.scatter(dataset_x, dataset_y, color='blue')
plt.plot(x, y, color='red')
plt.xticks(range(1, 15))
plt.show()

predict_data = 12.9
predict_result = b0 + predict_data * b1
print(predict_result)