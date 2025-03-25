import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

dataset_x = df.iloc[:, 2:-1].to_numpy()
dataset_y = np.zeros(len(df))
dataset_y[df['diagnosis'] == 'M'] = 1

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=12345)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(training_dataset_x)
scaled_training_dataset_x = ss.transform(training_dataset_x)
scaled_test_dataset_x = ss.transform(test_dataset_x)

# SVC Solution

from sklearn.svm import SVC

svc = SVC(kernel='linear')
svc.fit(scaled_training_dataset_x, training_dataset_y)

# LogisticRegression Solution

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000000)
lr.fit(scaled_training_dataset_x, training_dataset_y)

# Naive Bayes Solution

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(training_dataset_x, training_dataset_y)

# Neural Net Solution

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential(name='BreastCancer')
model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(64, activation='relu', name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size=32, epochs=200, verbose=0)

from sklearn.metrics import accuracy_score

predict_result_svc = svc.predict(scaled_test_dataset_x)
score_svc = accuracy_score(test_dataset_y, predict_result_svc)

predict_result_lr = lr.predict(scaled_test_dataset_x)
score_lr = accuracy_score(test_dataset_y, predict_result_lr)

predict_result_gnb = gnb.predict(test_dataset_x)
score_gnb = accuracy_score(test_dataset_y, predict_result_gnb)

predict_result_nn = (model.predict(scaled_test_dataset_x) > 0.5).astype(int)
score_nn = accuracy_score(test_dataset_y, predict_result_nn)

print(f'SVC accuracy score: {score_svc}')
print(f'LogisticRegression accuracy score: {score_lr}')
print(f'GaussianNB accuracy score: {score_gnb}')
print(f'NeuralNet accuracy score: {score_nn}')