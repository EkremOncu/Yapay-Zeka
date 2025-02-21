import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

dataset_x = df.iloc[:, 2:-1].to_numpy()
dataset_y = np.zeros(len(df))
dataset_y[df['diagnosis'] == 'M'] = 1

"""
from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
"""

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(training_dataset_x, training_dataset_y)
test_result = gnb.predict(test_dataset_x)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(test_dataset_y, test_result)
print(accuracy)
