import pandas as pd
import numpy as np

df = pd.read_csv('auto-mpg.data', delimiter=r'\s+', header=None)

df = df.iloc[:, :-1]
df.iloc[df.loc[:, 3] == '?', 3] = np.nan

df[3] = df[3].astype('float64')


from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean', missing_values=np.nan)
df[3] = si.fit_transform(df[[3]])

df = pd.get_dummies(df, columns=[7], dtype='uint8')

dataset = df.to_numpy()

dataset_x = dataset[:, 1:]
dataset_y = dataset[:, 0]


from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

        
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(training_dataset_x)
scaled_training_dataset_x = ss.transform(training_dataset_x)
scaled_test_dataset_x = ss.transform(test_dataset_x)
       

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential(name='Auto-MPG')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(32, activation='relu', name='Hidden-1'))
model.add(Dense(32, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='linear', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size=32, 
                 epochs=200, validation_split=0.2)

eval_result = model.evaluate(scaled_test_dataset_x, test_dataset_y, batch_size=32)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Epoch - Mean Absolute Error', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['mae'])
plt.plot(hist.epoch, hist.history['val_mae'])
plt.legend(['Measn Absolute Error', 'Validation Mean Absolute Error'])
plt.show()

# prediction

predict_df = pd.read_csv('predict.csv', header=None)
predict_df = pd.get_dummies(predict_df, columns=[6])

predict_dataset_x = predict_df.to_numpy() 
scaled_predict_dataset_x = ss.transform(predict_dataset_x)

predict_result = model.predict(scaled_predict_dataset_x)

for val in predict_result[:, 0]:
    print(val)
    
#one_hot_layer = tf.keras.layers.Lambda(lambda x: tf.one_hot(x, num_unique_chars))






