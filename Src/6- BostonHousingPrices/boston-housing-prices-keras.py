from tensorflow.keras.datasets import boston_housing

(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) = boston_housing.load_data()

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, dtype='uint8')

ohe_result_training = ohe.fit_transform(training_dataset_x[:, 8].reshape(-1, 1))
ohe_result_test = ohe.transform(test_dataset_x[:, 8].reshape(-1, 1))

import numpy as np

training_dataset_x = np.delete(training_dataset_x, 8, axis=1)
test_dataset_x = np.delete(test_dataset_x, 8, axis=1)

training_dataset_x = np.insert(training_dataset_x, [8], ohe_result_training, axis=1)
test_dataset_x = np.insert(test_dataset_x, [8], ohe_result_test, axis=1)


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(training_dataset_x)
scaled_training_dataset_x = ss.transform(training_dataset_x)
scaled_test_dataset_x = ss.transform(test_dataset_x)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential(name='Boston-Housing-Prices')
model.add(Input((training_dataset_x.shape[1], ), name='Input'))
model.add(Dense(64, activation='relu', name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='linear', name='Output'))
model.summary()

model.compile('rmsprop', loss='mse', metrics=['mae'])
hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size=32, epochs=200, validation_split=0.2)

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', fontsize=14, pad=10)
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Mean Absolute Error - Validation Mean Absolute Error Graph', fontsize=14, pad=10)
plt.plot(hist.epoch, hist.history['mae'])
plt.plot(hist.epoch, hist.history['val_mae'])
plt.legend(['Mean Absolute Error', 'Validation Mean Absolute Error'])
plt.show()

eval_result = model.evaluate(scaled_test_dataset_x , test_dataset_y, batch_size=32)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

"""
import pickle

model.save('boston-housing-prices.h5')
with open('boston-housing-prices.pickle', 'wb') as f:
    pickle.dump([ohe, ss], f)
"""

import pandas as pd

predict_df = pd.read_csv('predict-boston-housing-prices.csv', delimiter=r'\s+', header=None)
predict_dataset_x = predict_df.to_numpy()

ohe_result_predict = ohe.transform(predict_dataset_x [:, 8].reshape(-1, 1))

predict_dataset_x = np.delete(predict_dataset_x, 8, axis=1)
predict_dataset_x = np.insert(predict_dataset_x, [8], ohe_result_predict, axis=1)

scaled_predict_dataset_x = ss.transform(predict_dataset_x )
predict_result = model.predict(scaled_predict_dataset_x)

for val in predict_result[:, 0]:
    print(val)
    
    
    
    
    
    
    