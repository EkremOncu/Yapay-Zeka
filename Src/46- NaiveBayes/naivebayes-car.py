import pandas as pd

df = pd.read_csv('car.data', header=None)
df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

oe = OrdinalEncoder()
le = LabelEncoder()

encoded_dataset_x = oe.fit_transform(df.iloc[:, :-1].to_numpy(dtype='str'))
encoded_dataset_y = le.fit_transform(df.iloc[:, -1].to_numpy(dtype='str'))

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(encoded_dataset_x, encoded_dataset_y, test_size=0.2)

from sklearn.naive_bayes import CategoricalNB

cnb = CategoricalNB()

cnb.fit(training_dataset_x, training_dataset_y)

test_result = cnb.predict(test_dataset_x)

from sklearn.metrics import accuracy_score

score = accuracy_score(test_dataset_y, test_result)
print(score)

"""
import numpy as np

score = np.sum(test_dataset_y == test_result) / len(test_dataset_y)
print(score)
"""

predict_dataset_x = pd.read_csv('predict.csv', header=None).to_numpy(dtype='str')
encoded_predict_dataset_x = oe.transform(predict_dataset_x)

predict_result = cnb.predict(encoded_predict_dataset_x)
print(predict_result)
predict_result_names = le.inverse_transform(predict_result).tolist()

















