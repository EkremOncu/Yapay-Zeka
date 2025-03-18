import numpy as np
from sklearn.datasets import make_blobs

dataset_x, dataset_y = make_blobs(n_features=2, centers=3, cluster_std=2, random_state=100)

colors = ['red', 'green', 'blue']

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

for i in range(3):
    plt.scatter(dataset_x[dataset_y == i, 0], dataset_x[dataset_y == i, 1], color=colors[i])
plt.show()

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(dataset_x, dataset_y)

x = np.linspace(-15, 15, 300)

plt.figure(figsize=(12, 8))

plt.xlim(-20, 20)
plt.ylim(-20, 20)

for i in range(len(lr.coef_)):
    plt.scatter(dataset_x[dataset_y == i, 0], dataset_x[dataset_y == i, 1], color=colors[i])
    y = (-lr.intercept_[i] - lr.coef_[i, 0] * x) / lr.coef_[i, 1]
    plt.plot(x, y, color=colors[i])

plt.show()

print(lr.score(dataset_x, dataset_y))

lr = LogisticRegression(multi_class='ovr')
lr.fit(dataset_x, dataset_y)

x = np.linspace(-15, 15, 300)

plt.figure(figsize=(12, 8))

plt.xlim(-20, 20)
plt.ylim(-20, 20)

for i in range(len(lr.coef_)):
    plt.scatter(dataset_x[dataset_y == i, 0], dataset_x[dataset_y == i, 1], color=colors[i])
    y = (-lr.intercept_[i] - lr.coef_[i, 0] * x) / lr.coef_[i, 1]
    plt.plot(x, y, color=colors[i])

plt.show()

print(lr.score(dataset_x, dataset_y))






