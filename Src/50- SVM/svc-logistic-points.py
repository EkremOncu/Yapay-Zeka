import numpy as np
import pandas as pd

df = pd.read_csv('logistic-points.csv')

dataset_x = df.iloc[:, :-1].to_numpy()
dataset_y = df.iloc[:, -1].to_numpy()

import matplotlib.pyplot as plt

plt.figure(figsize=(9, 7))
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

from sklearn.svm import SVC

svc = SVC(kernel='linear')
svc.fit(scaled_dataset_x, dataset_y)

x = np.linspace(-3, 2, 1000)
y = -(svc.intercept_ + svc.coef_[0, 0] * x) / svc.coef_[0, 1]

points = np.vstack((x, x)).T
transformed_points = ss.inverse_transform(points)

support_x = dataset_x[svc.support_]
support_y = dataset_y[svc.support_]

import matplotlib.pyplot as plt

plt.figure(figsize=(9, 7))
plt.scatter(dataset_x[dataset_y == 0, 0], dataset_x[dataset_y == 0, 1], marker='o')
plt.scatter(dataset_x[dataset_y == 1, 0], dataset_x[dataset_y == 1, 1], marker='^')
plt.plot(transformed_points[:, 0], transformed_points[:, 1], color='red')
plt.scatter(support_x[support_y == 0, 0], support_x[support_y == 0, 1], marker='o', color='red')
plt.scatter(support_x[support_y == 1, 0], support_x[support_y == 1, 1], marker='^', color='red')
plt.legend(['Class-1', 'Class-2', 'Support Vector Line', 'Support Vectors Class1', 'Support Vectors Class2'])
plt.show()
     
print(svc.support_)      
print(svc.support_vectors_)      

predict_data = [(-0.733928, 9.098687) , (0-3.642001, -1.618087), (0.556921, 8.294984)]
scaled_predict_data = ss.transform(predict_data)

predict_result = svc.predict(scaled_predict_data)
print(predict_result)



