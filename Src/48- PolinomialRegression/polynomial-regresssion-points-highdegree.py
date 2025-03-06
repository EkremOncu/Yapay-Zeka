import numpy as np
import pandas as pd

df = pd.read_csv('points.csv')

dataset_x = df['X'].to_numpy()
dataset_y = df['Y'].to_numpy()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

degrees = [2, 3, 4, 5]
rsquares = []

plt.figure(figsize=(10, 8))
plt.title('Polynomial Regression')
plt.scatter(dataset_x, dataset_y, color='blue')
plt.ylim(-500, 3000)

for i in degrees:  
    pf = PolynomialFeatures(degree=i)
    transformed_dataset_x = pf.fit_transform(dataset_x.reshape(-1, 1))
    lr = LinearRegression()
    lr.fit(transformed_dataset_x, dataset_y)
    
    x = np.linspace(-5, 20, 1000)
    transformed_x = pf.transform(x.reshape(-1, 1))
    y = lr.predict(transformed_x)
    plt.plot(x, y)
    
    rsquares.append(lr.score(transformed_dataset_x, dataset_y))
    
plt.legend( ['points'] + [f'{degree} ({rs:.2f})' for degree, rs in zip(degrees, rsquares)])
plt.show()






