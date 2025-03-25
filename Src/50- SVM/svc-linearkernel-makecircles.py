import numpy as np
from sklearn.datasets import make_circles

dataset_x, dataset_y = make_circles(n_samples=100, noise=0.01)

from sklearn.svm import SVC

svc = SVC(kernel='linear')
svc.fit(dataset_x, dataset_y)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(dataset_x, dataset_y)

x = np.linspace(np.min(dataset_x[:, 0]), np.max(dataset_x[:, 0]), 1000)
y_svc = -(svc.intercept_ + svc.coef_[0, 0] * x) / svc.coef_[0, 1]
y_logistic = -(lr.intercept_ + lr.coef_[0, 0] * x) / lr.coef_[0, 1]

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(dataset_x[dataset_y == 0, 0], dataset_x[dataset_y == 0, 1], marker='o')
plt.scatter(dataset_x[dataset_y == 1, 0], dataset_x[dataset_y == 1, 1], marker='^')
plt.plot(x, y_svc)
plt.plot(x, y_logistic)
plt.legend(['Class-1', 'Class-2', 'Support Vector Machine', 'Logistic Regression'])
plt.show()
           
svc_score = svc.score(dataset_x, dataset_y)
logistic_score = lr.score(dataset_x, dataset_y)

print(f'SVC score: {svc_score}')
print(f'Logistic score: {logistic_score}')