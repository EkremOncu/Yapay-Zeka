import numpy as np
import pandas as pd

df = pd.read_csv('logistic-points.csv')

dataset_x = df.iloc[:, :-1].to_numpy()
dataset_y = df.iloc[:, -1].to_numpy()

from sklearn.svm import SVC

svc = SVC(kernel='linear')
svc.fit(dataset_x, dataset_y)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(dataset_x, dataset_y)

x = np.linspace(-4, 4, 1000)
y_svc = -(svc.intercept_ + svc.coef_[0, 0] * x) / svc.coef_[0, 1]
y_logistic = -(lr.intercept_ + lr.coef_[0, 0] * x) / lr.coef_[0, 1]

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(dataset_x[dataset_y == 0, 0], dataset_x[dataset_y == 0, 1], marker='o')
plt.scatter(dataset_x[dataset_y == 1, 0], dataset_x[dataset_y == 1, 1], marker='^')
plt.plot(x, y_svc)
plt.plot(x, y_logistic)

plt.legend(['Class-1', 'Class-2', 'Support Vector Machine', 'Logistic Regression', "Support Vectors"])
plt.show()
           
svc_score = svc.score(dataset_x, dataset_y)
logistic_score = lr.score(dataset_x, dataset_y)

print(f'SVC score: {svc_score}')
print(f'Logistic score: {logistic_score}')