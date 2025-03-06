import numpy as np
import pandas as pd

df = pd.read_csv('points.csv')

dataset_x = df['X'].to_numpy()
dataset_y = df['Y'].to_numpy()

from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=3)

transformed_dataset_x = pf.fit_transform(dataset_x.reshape(-1, 1))

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(transformed_dataset_x, dataset_y)

rsquare = lr.score(transformed_dataset_x, dataset_y)
print(f'R^2: {rsquare}')

x = np.linspace(-5, 20, 1000)
transformed_x = pf.transform(x.reshape(-1, 1))
y = lr.predict(transformed_x)

import matplotlib.pyplot as plt

plt.title('Polynomial Regression')
plt.scatter(dataset_x, dataset_y, color='blue')
plt.plot(x, y, color='red')
plt.show()

predict_data = np.array([12.9, 4.7, 6.9])
predict_result = lr.intercept_ + lr.coef_[1] * predict_data  + lr.coef_[2] * predict_data ** 2 + lr.coef_[3] * predict_data ** 3
print(predict_result)

predict_data = np.array([[12.9], [4.7], [6.9]])
transformed_predict_data = pf.transform(predict_data)
predict_result = lr.predict(transformed_predict_data)
print(predict_result)



