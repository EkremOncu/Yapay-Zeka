import pandas as pd
import numpy as np


PREDICTION_INTERVAL = 24 * 60 // 10         # 144
WINDOW_SIZE = 24 * 60 // 10                 # 144
SLIDING_SIZE = 5


df = pd.read_csv('jena_climate_2009_2016.csv')


df['Month'] = df['Date Time'].str[3:5]
df['Hour-Minute'] = df['Date Time'].str[11:16]


df.drop(['Date Time'], axis=1, inplace=True)


from sklearn.preprocessing import OneHotEncoder


ohe = OneHotEncoder(sparse_output=False)


ohe.fit(df[['Month', 'Hour-Minute']])
ohe_result = ohe.transform(df[['Month', 'Hour-Minute']])


df = pd.concat([df, pd.DataFrame(ohe_result)], axis=1)
df.drop(['Month', 'Hour-Minute'], axis=1, inplace=True)


raw_dataset_x = df.to_numpy('float32') 
raw_dataset_y = df['T (degC)'].to_numpy('float32')


from sklearn.model_selection import train_test_split


raw_training_dataset_x, raw_test_dataset_x, raw_training_dataset_y, raw_test_dataset_y = train_test_split(raw_dataset_x, raw_dataset_y, test_size=0.2, shuffle=False)


from sklearn.preprocessing import StandardScaler


ss = StandardScaler()
ss.fit(raw_training_dataset_x)

raw_scaled_training_dataset_x = ss.transform(raw_training_dataset_x)
raw_scaled_test_dataset_x = ss.transform(raw_test_dataset_x)


def create_ts_dataset(dataset_x, dataset_y, pi, ws, ss):
    x = []
    y = []
    for i in range(0, len(dataset_x) - ws - pi, ss):
        x.append(dataset_x[i:i + ws])
        y.append(dataset_y[i + ws + pi - 1])


    return np.array(x), np.array(y)


scaled_training_dataset_x, training_dataset_y = create_ts_dataset(raw_scaled_training_dataset_x, raw_training_dataset_y, PREDICTION_INTERVAL, WINDOW_SIZE, SLIDING_SIZE)


scaled_test_dataset_x, test_dataset_y = create_ts_dataset(raw_scaled_test_dataset_x, raw_test_dataset_y, PREDICTION_INTERVAL, WINDOW_SIZE, SLIDING_SIZE)


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Reshape, Dense


model = Sequential(name='Jena-Climate')
model.add(Input((scaled_training_dataset_x.shape[1], scaled_training_dataset_x.shape[2]),  name='Input'))


model.add(Conv1D(128, 3, padding='same', name='Conv1D-1'))
model.add(MaxPooling1D(2, padding='same', name='MaxPooling1D-1'))

model.add(Conv1D(128, 3, padding='same', name='Conv1D-2'))
model.add(MaxPooling1D(2, padding='same', name='MaxPooling1D-2'))

model.add(Conv1D(128, 3, padding='same', name='Conv1D-3'))
model.add(MaxPooling1D(2, padding='same', name='MaxPooling1D-3'))

model.add(Reshape((-1, ), name='Reshape'))

model.add(Dense(256, activation='relu', name='Hidden-1'))
model.add(Dense(256, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='linear', name='Output'))
model.summary()


model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])


from tensorflow.keras.callbacks import EarlyStopping


esc = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)


hist = model.fit(scaled_training_dataset_x, training_dataset_y, epochs=100, batch_size=32, 
                 validation_split=0.2, callbacks=[esc])


import matplotlib.pyplot as plt


plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()


plt.figure(figsize=(14, 6))
plt.title('Epoch - Mean Squared Error', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['mean_absolute_error'])
plt.plot(hist.epoch, hist.history['val_mean_absolute_error'])
plt.legend(['MSE', 'Validation MSE'])
plt.show()


# evaluation


eval_result = model.evaluate(scaled_test_dataset_x, test_dataset_y)


for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')


# prediction


predict_df = pd.read_csv('predict.csv')


predict_df['Month'] = predict_df['Date Time'].str[3:5]
predict_df['Hour-Minute'] = predict_df['Date Time'].str[11:16]


predict_df.drop(['Date Time'], axis=1, inplace=True)


ohe_result = ohe.transform(predict_df[['Month', 'Hour-Minute']])


predict_df = pd.concat([predict_df, pd.DataFrame(ohe_result)], axis=1)
predict_df.drop(['Month', 'Hour-Minute'], axis=1, inplace=True)


predict_dataset = predict_df.to_numpy('float32')
scaled_predict_dataset = ss.transform(predict_dataset)


predict_result = model.predict(scaled_predict_dataset.reshape(1, predict_dataset.shape[0], predict_dataset.shape[1]))


for presult in predict_result[:, 0]:
    print(presult)


