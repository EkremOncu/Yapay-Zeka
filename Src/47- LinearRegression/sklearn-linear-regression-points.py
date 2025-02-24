import numpy as np
import pandas as pd

df = pd.read_csv('points.csv')

dataset_x = df['x'].to_numpy()
dataset_y = df['y'].to_numpy()

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(dataset_x.reshape(-1, 1), dataset_y)

x = np.linspace(0, 15, 1000)
y = lr.intercept_ + lr.coef_[0] * x

import matplotlib.pyplot as plt

plt.title('Simple Linear Regression')
plt.scatter(dataset_x, dataset_y, color='blue')
plt.plot(x, y, color='red')
plt.xticks(range(1, 15))
plt.show()

predict_data = 12.9
predict_result = lr.intercept_ + predict_data * lr.coef_[0]
print(predict_result)

predict_data = np.array([[12.9], [4.7], [6.9]])
predict_result = lr.predict(predict_data)
print(predict_result)

rsquare = lr.score(dataset_x.reshape(-1, 1), dataset_y)
print(f'R^2: {rsquare}')