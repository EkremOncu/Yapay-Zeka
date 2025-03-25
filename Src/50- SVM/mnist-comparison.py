import pandas as pd

df_training = pd.read_csv('mnist_train.csv')
df_test = pd.read_csv('mnist_test.csv')

training_dataset_x = df_training.iloc[:, 1:].to_numpy()
training_dataset_y = df_training.iloc[:, 0].to_numpy()

test_dataset_x = df_test.iloc[:, 1:].to_numpy()
test_dataset_y = df_test.iloc[:, 0].to_numpy()

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(training_dataset_x, training_dataset_y)
predict_result = lr.predict(test_dataset_x)

from sklearn.metrics import accuracy_score

score = accuracy_score(test_dataset_y, predict_result)
print(f'LogisticRegresson accuracy score: {score}')

from tensorflow.keras.utils import to_categorical

ohe_training_dataset_y = to_categorical(training_dataset_y)
ohe_test_dataset_y = to_categorical(test_dataset_y)

training_dataset_x = training_dataset_x / 255
test_dataset_x = test_dataset_x / 255

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential(name='MNIST') 
model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(256, activation='relu', name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(10, activation='softmax', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
hist = model.fit(training_dataset_x, ohe_training_dataset_y, epochs=20, batch_size=32)

import numpy as np

softmax_predict_result = model.predict(test_dataset_x)
predict_result = np.argmax(softmax_predict_result, axis=1)

score = accuracy_score(test_dataset_y, predict_result)
print(f'Simple Neural Net accuracy score: {score}')

from sklearn.svm import SVC

svc = SVC(kernel='rbf')
svc.fit(training_dataset_x, training_dataset_y)

predict_result = svc.predict(test_dataset_x)
score = accuracy_score(test_dataset_y, predict_result)
print(f'SVC accuracy score: {score}')












