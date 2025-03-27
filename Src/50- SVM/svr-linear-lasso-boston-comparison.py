import pandas as pd

df = pd.read_csv('housing.csv', delimiter=r'\s+', header=None)

dataset_x = df.iloc[:, :-1].to_numpy(dtype='float32')
dataset_y = df.iloc[:, -1].to_numpy(dtype='float32')

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=1234)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(training_dataset_x)

scaled_training_dataset_x = ss.transform(training_dataset_x)
scaled_test_dataset_x = ss.transform(test_dataset_x)

from sklearn.svm import SVR

svr = SVR(kernel='linear')
svr.fit(scaled_training_dataset_x, training_dataset_y)
predict_result = svr.predict(scaled_test_dataset_x)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(predict_result, test_dataset_y)
print(f'SVR Mean Absolute Error: {mae}')

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(training_dataset_x, training_dataset_y)
predict_result = lr.predict(test_dataset_x)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(predict_result, test_dataset_y)
print(f'LinearRegression Mean Absolute Error: {mae}')

from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2)
transformed_training_dataset_x = pf.fit_transform(training_dataset_x)

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.005, max_iter=100000)
lasso.fit(scaled_training_dataset_x , training_dataset_y)
predict_result = lasso.predict(scaled_test_dataset_x)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(predict_result, test_dataset_y)
print(f'Lasso Polynomial Mean Absolute Error: {mae}')

















