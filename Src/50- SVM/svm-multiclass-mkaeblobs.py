from sklearn.datasets import make_blobs

dataset_x, dataset_y = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=5, random_state=12345)

from sklearn.svm import SVC

svc = SVC(kernel='rbf')
svc.fit(dataset_x, dataset_y)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(dataset_x, dataset_y)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
plt.scatter(dataset_x[dataset_y == 0, 0], dataset_x[dataset_y == 0, 1], marker='o')
plt.scatter(dataset_x[dataset_y == 1, 0], dataset_x[dataset_y == 1, 1], marker='^')
plt.scatter(dataset_x[dataset_y == 2, 0], dataset_x[dataset_y == 2, 1], marker='v')
plt.legend(['Class-1', 'Class-2', 'Class-3'])
plt.show()
           
svc_score = svc.score(dataset_x, dataset_y)
logistic_score = lr.score(dataset_x, dataset_y)

print(f'SVC score: {svc_score}')
print(f'Logistic score: {logistic_score}')
