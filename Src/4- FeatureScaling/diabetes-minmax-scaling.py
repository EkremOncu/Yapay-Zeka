import pandas as pd

df = pd.read_csv('diabetes.csv')

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean', missing_values=0)

impute_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[impute_features] = si.fit_transform(df[impute_features])

dataset = df.to_numpy()

dataset_x = dataset[:, :-1]
dataset_y = dataset[:, -1]

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)


# ---------------------------------------

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

mms.fit(training_dataset_x)
scaled_training_dataset_x = mms.transform(training_dataset_x)
scaled_test_dataset_x = mms.transform(test_dataset_x)

# ---------------------------------------

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential(name='Diabetes')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(16, activation='relu', name='Hidden-1'))
model.add(Dense(16, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size=32, epochs=100, validation_split=0.2)
eval_result = model.evaluate(scaled_test_dataset_x, test_dataset_y, batch_size=32)


import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Epoch - Binary Accuracy Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['binary_accuracy'])
plt.plot(hist.epoch, hist.history['val_binary_accuracy'])
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.show()

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

import numpy as np

predict_dataset = np.array([[2 ,90, 68, 12, 120, 38.2, 0.503, 28],
                            [4, 111, 79, 47, 207, 37.1, 1.39, 56],
                            [3, 190, 65, 25, 130, 34, 0.271, 26],
                            [8, 176, 90, 34, 300, 50.7, 0.467, 58],
                            [7, 106, 92, 18, 200, 35, 0.300, 48]])

# ---------------------------------------

scaled_predict_dataset = mms.transform(predict_dataset)
predict_result = model.predict(scaled_predict_dataset)
print(predict_result)

# ---------------------------------------

for result in predict_result[:, 0]:
    print('Şeker hastası' if result > 0.5 else 'Şeker Hastası Değil')

