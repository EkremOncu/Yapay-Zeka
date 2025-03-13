import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

dataset_x = df.iloc[:, 2:-1].to_numpy()
dataset_y = np.zeros(len(df))
dataset_y[df['diagnosis'] == 'M'] = 1

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=12345)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

ss = StandardScaler()
ss.fit(training_dataset_x)
scaled_training_dataset_x = ss.transform(training_dataset_x)
scaled_test_dataset_x = ss.transform(test_dataset_x)

# LogisticRegression Solution

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000000)
lr.fit(scaled_training_dataset_x, training_dataset_y)
predict_result = lr.predict(scaled_test_dataset_x)
score = accuracy_score(test_dataset_y, predict_result)
print(f'LogisticRegression accuracy score: {score}')

# Naive Bayes Solution

gnb = GaussianNB()
gnb.fit(training_dataset_x, training_dataset_y)
predict_result = gnb.predict(test_dataset_x)
score = accuracy_score(predict_result, test_dataset_y)
print(f'GaussianNB accuracy score: {score}')

# Neural Net Solution

model = Sequential(name='BreastCancer')
model.add(Dense(64, activation='relu', input_dim=dataset_x.shape[1], name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size=32, epochs=200, verbose=0)
predict_result = (model.predict(scaled_test_dataset_x) > 0.5).astype(int)
score = accuracy_score(predict_result, test_dataset_y)
print(f'NeuralNet accuracy score: {score}')


