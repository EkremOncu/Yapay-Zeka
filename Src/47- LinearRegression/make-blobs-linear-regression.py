import numpy as np
from sklearn.datasets import make_blobs

dataset, _ = make_blobs(n_samples=100, centers=1, center_box=(0, 0))

dataset_x = dataset[:, 0].reshape(-1, 1)
dataset_y = dataset[:, 1]

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(dataset_x, dataset_y)

x = np.linspace(-5, 5, 100)
y = lr.predict(x.reshape(-1, 1))

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.title('Linear Regression with Circuler Data')
plt.scatter(dataset_x, dataset_y, color='blue')
plt.plot(x, y, color='red')
plt.show()

r2 = lr.score(dataset_x, dataset_y)
print(f'R^2 = {r2}')

cor = np.corrcoef(dataset_x.flatten(), dataset_y)
print(cor)