from tensorflow.keras.datasets import mnist

(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) = mnist.load_data()

training_dataset_x = training_dataset_x.reshape(-1, 28 * 28)
test_dataset_x = test_dataset_x.reshape(-1, 28 * 28)

# one hot encoding for y data

from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

ohe_training_dataset_y = to_categorical(training_dataset_y)
ohe_test_dataset_y = to_categorical(test_dataset_y)

# minmax scaling

scaled_training_dataset_x = training_dataset_x / 255
scaled_test_dataset_x = test_dataset_x / 255

# LogisticRegression Solution

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(training_dataset_x, training_dataset_y)
predict_result = lr.predict(test_dataset_x)
score = accuracy_score(test_dataset_y, predict_result)
print(f'LogisticRegresson accuracy score: {score}')

# Simple Neural Net Solution

from tensorflow.keras.utils import to_categorical

ohe_training_dataset_y = to_categorical(training_dataset_y)
ohe_test_dataset_y = to_categorical(test_dataset_y)

training_dataset_x = training_dataset_x / 255
test_dataset_x = test_dataset_x / 255

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='MNIST') 
model.add(Dense(256, activation='relu', input_dim=784, name='Hidden-1'))
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