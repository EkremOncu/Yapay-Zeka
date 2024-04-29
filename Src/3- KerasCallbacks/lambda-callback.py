import pandas as pd

df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\GitHub\\YapayZeka\\Src\\2- KerasIntroduction\diabetes.csv')

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean', missing_values=0)

df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = si.fit_transform(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])

dataset = df.to_numpy()

dataset_x = dataset[:, :-1]
dataset_y = dataset[:, -1]

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential(name='Diabetes')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(16, activation='relu', name='Hidden-1'))
model.add(Dense(16, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])

import numpy as np

batch_losses= []

def on_epoch_begin_proc(epoch, logs):
    global batch_losses
    batch_losses = []
    print(f'eopch: {epoch}')    

def on_epoch_end_proc(epoch, logs):
    loss = logs['loss']
    val_loss = logs['val_loss']
    print(f'\nepoch: {epoch}, loss: {loss}, val_loss: {val_loss}')
    print(f'batch mean: {np.mean(batch_losses)}')
    print('-' * 30)
  
def on_batch_end_proc(batch, logs):
    global total
    loss = logs['loss']
    batch_losses.append(loss)
    print(f'\t\tbatch: {batch}, loss: {loss}')
    
from tensorflow.keras.callbacks import LambdaCallback

lambda_callback = LambdaCallback(on_epoch_begin=on_epoch_begin_proc, on_epoch_end=on_epoch_end_proc,  
                                 on_batch_end=on_batch_end_proc)

hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=100, 
                 validation_split=0.2, callbacks=[lambda_callback], verbose=0)

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

eval_result = model.evaluate(test_dataset_x, test_dataset_y, batch_size=32)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
    













