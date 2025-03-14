import numpy as np
import pandas as pd

df = pd.read_csv('logistic-points.csv')

dataset_x = df.iloc[:, :-1].to_numpy()
dataset_y = df.iloc[:, -1].to_numpy()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.title('Points for Logistic Regression')
plt.scatter(dataset_x[dataset_y == 0, 0], dataset_x[dataset_y == 0, 1], color='blue', marker='o')
plt.scatter(dataset_x[dataset_y == 1, 0], dataset_x[dataset_y == 1, 1], color='green', marker='^')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(['class 0', 'class 1'])

plt.show()

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
scaled_dataset_x = ss.fit_transform(dataset_x)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(scaled_dataset_x, dataset_y)

x1 = np.linspace(-5, 5, 1000)
x2 = -(lr.intercept_ + lr.coef_[0, 0] * x1) / lr.coef_[0, 1]

plt.figure(figsize=(10, 8))
plt.title('Points for Logistic Regression')
plt.scatter(scaled_dataset_x[dataset_y == 0, 0], scaled_dataset_x[dataset_y == 0, 1], color='blue', marker='o')
plt.scatter(scaled_dataset_x[dataset_y == 1, 0], scaled_dataset_x[dataset_y == 1, 1], color='green', marker='^')
plt.plot(x1, x2, color='red')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(['class 0', 'class 1'])
plt.show()

# inverse_tramsform 

x1 = np.linspace(-5, 5, 1000)
x2 = -(lr.intercept_ + lr.coef_[0, 0] * x1) / lr.coef_[0, 1]

points = np.vstack((x1, x2)).T
transformed_points = ss.inverse_transform(points)

plt.figure(figsize=(10, 8))
plt.title('Points for Logistic Regression')
plt.scatter(dataset_x[dataset_y == 0, 0], dataset_x[dataset_y == 0, 1], color='blue', marker='o')
plt.scatter(dataset_x[dataset_y == 1, 0], dataset_x[dataset_y == 1, 1], color='green', marker='^')
plt.plot(transformed_points[:, 0], transformed_points[:, 1], color='red')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(['class 0', 'class 1'])
plt.show()

predict_data = [(-0.733928, 9.098687) , (0-3.642001, -1.618087), (0.556921, 8.294984)]

scaled_predict_data = ss.transform(predict_data)

predict_result = lr.predict(scaled_predict_data)
print(predict_result)

predict_proba_result = lr.predict_proba(scaled_predict_data)
print(predict_proba_result)







