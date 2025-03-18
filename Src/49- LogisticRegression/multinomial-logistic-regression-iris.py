import pandas as pd
    
df = pd.read_csv('iris.csv')

dataset_x = df.iloc[:, 1:-1].to_numpy()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
dataset_y = le.fit_transform(df.iloc[:, -1])

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=12345)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(training_dataset_x)
scaled_training_dataset_x = ss.transform(training_dataset_x)
scaled_test_dataset_x = ss.transform(test_dataset_x)

# LogisticRegression Solution

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression()
lr.fit(scaled_training_dataset_x, training_dataset_y)

predict_result = lr.predict(scaled_test_dataset_x)
score = accuracy_score(test_dataset_y, predict_result)
print(f'Multinomial LogisticRegression score: {score}')

# Neural Net Solution

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)
ohe_training_dataset_y = ohe.fit_transform(training_dataset_y.reshape(-1, 1))
ohe_test_dataset_y = ohe.fit_transform(test_dataset_y.reshape(-1, 1))

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='Iris')
model.add(Dense(64, activation='relu', input_dim=dataset_x.shape[1], name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(3, activation='softmax', name='Output'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
hist = model.fit(scaled_training_dataset_x, ohe_training_dataset_y, batch_size=32, epochs=200)
predict_result_softmax = model.predict(scaled_test_dataset_x)

import numpy as np

predict_result = np.argmax(predict_result_softmax, axis=1)
score = accuracy_score(test_dataset_y, predict_result)
print(f'Neural Net score: {score}')