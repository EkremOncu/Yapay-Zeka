import pandas as pd

df = pd.read_csv('housing.csv', delimiter=r'\s+', header=None)

dataset_x = df.iloc[:, :-1].to_numpy(dtype='float32')
dataset_y = df.iloc[:, -1].to_numpy(dtype='float32')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=1234)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(training_dataset_x, training_dataset_y)
predict_result = lr.predict(test_dataset_x)

mae = mean_absolute_error(predict_result, test_dataset_y)
print(f'Mean Absolute Error: {mae}')
r2 = lr.score(test_dataset_x, test_dataset_y)
print(f'R^2 = {r2}')
print()

from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2)
transformed_training_dataset_x = pf.fit_transform(training_dataset_x)

lr = LinearRegression()
lr.fit(transformed_training_dataset_x , training_dataset_y)
transformed_test_dataset_x = pf.transform(test_dataset_x)
predict_result = lr.predict(transformed_test_dataset_x)

mae = mean_absolute_error(predict_result, test_dataset_y)
print(f'Mean Absolute Error: {mae}')
r2 = lr.score(transformed_test_dataset_x, test_dataset_y)
print(f'R^2 = {r2}')
print()

ss = StandardScaler()
scaled_transformed_training_dataset_x = ss.fit_transform(transformed_training_dataset_x)
scaled_transformed_test_dataset_x = ss.transform(transformed_test_dataset_x)

lasso = Lasso(alpha=0.005, max_iter=100000)
lasso.fit(scaled_transformed_training_dataset_x , training_dataset_y)
predict_result = lasso.predict(scaled_transformed_test_dataset_x)

mae = mean_absolute_error(predict_result, test_dataset_y)
print(f'Mean Absolute Error: {mae}')
r2 = lasso.score(scaled_transformed_test_dataset_x, test_dataset_y)
print(f'R^2 = {r2}')
print()