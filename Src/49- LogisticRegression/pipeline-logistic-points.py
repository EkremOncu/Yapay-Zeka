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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pl = Pipeline([('scaling', StandardScaler()), ('logistic-regression', LogisticRegression())])

pl.fit(dataset_x, dataset_y)

predict_data = [(-0.733928, 9.098687) , (0-3.642001, -1.618087), (0.556921, 8.294984)]
predict_result = pl.predict(predict_data)
print(predict_result)




