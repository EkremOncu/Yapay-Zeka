import pandas as pd

df = pd.read_csv('diabetes.csv')

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean', missing_values=0)

df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = si.fit_transform(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])

dataset = df.to_numpy()

dataset_x = dataset[:, :-1]
dataset_y = dataset[:, -1]

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

print(type(training_dataset_x))

# numpy'e döndürmeden (df.to_numpy()), direkt dataFrame ile de yapılabilir

dataset_x_df = df.iloc[:, :-1]
dataset_y_df = df.iloc[:, -1]

training_dataset_x_df, test_dataset_x_df, training_dataset_y_df, test_dataset_y_df = train_test_split(dataset_x_df, dataset_y_df, test_size=0.2)

print(type(training_dataset_x_df))