import pandas as pd

df = pd.read_csv('housing.csv', delimiter=r'\s+', header=None)

dataset_x = df.iloc[:, :-1].to_numpy(dtype='float32')
dataset_y = df.iloc[:, -1].to_numpy(dtype='float32')

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(training_dataset_x)
scaled_training_dataset_x = ss.transform(training_dataset_x)
scaled_test_dataset_x = ss.transform(test_dataset_x)

from sklearn.decomposition import PCA

pca = PCA(10)

pca.fit(scaled_training_dataset_x)
reduced_training_dataset_x = pca.transform(scaled_training_dataset_x)
reduced_test_dataset_x = pca.transform(scaled_test_dataset_x)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='Boston-Housing-Prices')
model.add(Dense(64, activation='relu', input_dim=reduced_training_dataset_x.shape[1], name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='linear', name='Output'))

model.summary()

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
hist = model.fit(reduced_training_dataset_x, training_dataset_y, batch_size=32, epochs=200, validation_split=0.2)

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.title('Epoch-Loss Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, 210, 10))

plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(15, 5))
plt.title('Epoch-Mean Absolute Error Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, 210, 10))

plt.plot(hist.epoch, hist.history['mae'])
plt.plot(hist.epoch, hist.history['val_mae'])
plt.legend(['Mean Absolute Error', 'Validation Mean Absolute Error'])
plt.show()

eval_result = model.evaluate(reduced_test_dataset_x, test_dataset_y)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
    
import numpy as np

predict_data = np.array([[0.11747, 12.50, 7.870, 0, 0.5240, 6.0090, 82.90, 6.2267, 5, 311.0, 15.20, 396.90, 13.27]])

scaled_predict_data = ss.transform(predict_data)
reduced_predict_data = pca.transform(scaled_predict_data)
predict_result = model.predict(reduced_predict_data)

for val in predict_result[:, 0]:
    print(val)
    
model.save('boston.h5')

import pickle

with open('boston.pickle', 'wb') as f:
    pickle.dump(ss, f)
    pickle.dump(pca, f)
    
