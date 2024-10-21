import pandas as pd
import numpy as np

PREDICTION_INTERVAL = 24 * 60 // 10         # 144
WINDOW_SIZE = 24 * 60 // 10                 # 144
SLIDING_SIZE = 5

BATCH_SIZE = 32
EPOCHS = 200

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

raw_temp_dataset_x, raw_test_dataset_x, raw_temp_dataset_y, raw_test_dataset_y =  train_test_split(raw_dataset_x, 
        raw_dataset_y, test_size=0.1, shuffle=False)

raw_training_dataset_x, raw_validation_dataset_x, raw_training_dataset_y, raw_validation_dataset_y =  train_test_split(raw_temp_dataset_x, 
        raw_temp_dataset_y, test_size=0.1, shuffle=False)


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(raw_training_dataset_x)
raw_scaled_training_dataset_x = ss.transform(raw_training_dataset_x)
raw_scaled_validation_dataset_x = ss.transform(raw_validation_dataset_x)
raw_scaled_test_dataset_x = ss.transform(raw_test_dataset_x)

import tensorflow as tf
from tensorflow.keras.utils import PyDataset

class DataGenerator(PyDataset):
    def __init__(self, raw_x, raw_y, batch_size, pi, ws, ss, *, shuffle=True):
        super().__init__() 
        self.raw_x = raw_x
        self.raw_y = raw_y
        self.batch_size = batch_size
        self.pi = pi
        self.ws = ws
        self.ss = ss
        self.shuffle = shuffle
        self.nbatches = (len(raw_x) - pi - ws) // batch_size // ss
        self.index_list = list(range((len(raw_x) - pi - ws) // ss))  
        
    def __len__(self):
        return self.nbatches
    
    def __getitem__(self, batch_no):               
        x = np.zeros((self.batch_size, self.ws, self.raw_x.shape[1]))
        y = np.zeros(self.batch_size)
        
        for i in range(self.batch_size):
            offset = self.index_list[batch_no * self.batch_size + i] * self.ss 
            
            x[i] = self.raw_x[offset:offset + self.ws]
            y[i] = self.raw_y[offset + self.ws + self.pi - 1]
     
        return tf.convert_to_tensor(x), tf.convert_to_tensor(y)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.index_list)      
   
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Reshape, Dense

model = Sequential(name='Jena-Climate')
model.add(Input((WINDOW_SIZE, raw_training_dataset_x.shape[1]),  name='Input'))

model.add(SimpleRNN(128, activation='tanh', return_sequences=True, name='SimpleRNN-1'))
model.add(SimpleRNN(128, activation='tanh', return_sequences=True, name='SimpleRNN-2'))
model.add(SimpleRNN(128, activation='tanh', return_sequences=True, name='SimpleRNN-3'))

model.add(Reshape((-1, ), name='Reshape'))

model.add(Dense(256, activation='relu', name='Hidden-1'))
model.add(Dense(256, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='linear', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

from tensorflow.keras.callbacks import EarlyStopping

esc = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

dg_training = DataGenerator(raw_scaled_training_dataset_x, raw_training_dataset_y, 
                            BATCH_SIZE, PREDICTION_INTERVAL, WINDOW_SIZE, SLIDING_SIZE, shuffle=False)

dg_validation = DataGenerator(raw_scaled_validation_dataset_x, raw_validation_dataset_y, 
                              BATCH_SIZE, PREDICTION_INTERVAL, WINDOW_SIZE, SLIDING_SIZE, shuffle=False)

hist = model.fit(dg_training, validation_data = dg_validation, epochs=EPOCHS, verbose=1, callbacks=[esc]) 
                 

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

dg_test = DataGenerator(raw_scaled_test_dataset_x, raw_test_dataset_y, BATCH_SIZE, PREDICTION_INTERVAL, WINDOW_SIZE, SLIDING_SIZE, shuffle=False)

eval_result = model.evaluate(dg_test)

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
