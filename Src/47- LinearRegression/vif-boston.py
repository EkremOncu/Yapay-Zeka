CORR_THREASHOLD = 0.45

import pandas as pd

df = pd.read_csv('housing.csv', delimiter=r'\s+', header=None)

dataset_x = df.iloc[:, :-1].to_numpy(dtype='float32')
dataset_y = df.iloc[:, -1].to_numpy(dtype='float32')

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=1234)

import numpy as np

concat_dataset = np.concatenate((training_dataset_x, training_dataset_y.reshape(-1, 1)), axis=1)
corr = np.abs(np.corrcoef(concat_dataset, rowvar=False))

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))
sns.heatmap(data=corr, annot=True)
plt.show()

selected_cols, = np.where(corr[:-1, -1] > CORR_THREASHOLD)
print(selected_cols)

from statsmodels.stats.outliers_influence import variance_inflation_factor

vifs =  np.array([variance_inflation_factor(training_dataset_x, i) for i in range(training_dataset_x.shape[1])])

for i, vif in enumerate(vifs):
    print(f'{i} ---> {vif}')

selected_cols, = np.where(vifs < 5)
print(selected_cols)

selected_training_dataset_x = training_dataset_x[:, selected_cols]
selected_test_dataset_x = test_dataset_x[:, selected_cols]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

lr = LinearRegression()
lr.fit(selected_training_dataset_x, training_dataset_y)

predict_result = lr.predict(selected_test_dataset_x)
mae = mean_absolute_error(predict_result, test_dataset_y)
print(f'Mean Absolute Error: {mae}')

r2 = lr.score(selected_test_dataset_x, test_dataset_y)
print(f'R^2 = {r2}')



