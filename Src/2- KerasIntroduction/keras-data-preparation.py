TRAINING_RATIO = 0.80

import pandas as pd

df = pd.read_csv('diabetes.csv')

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean', missing_values=0)

df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = si.fit_transform(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])

dataset = df.to_numpy()

import numpy as np

np.random.shuffle(dataset)

dataset_x = dataset[:, :-1]
dataset_y = dataset[:, -1]

training_len = int(np.round(len(dataset_x) * TRAINING_RATIO))

training_dataset_x = dataset_x[:training_len]
test_dataset_x = dataset_x[training_len:]

training_dataset_y = dataset_y[:training_len]
test_dataset_y = dataset_x[training_len:]




