import math
import random
import csv
import pandas as pd
import numpy as np

EPOCHS = 5
BATCH_SIZE = 32

def record_offsets(f, skiprows=1):
    offsets = []
    for i in range(skiprows):
        f.readline()
    while True:
        offsets.append(f.tell())
        if f.readline() != '':
            break  
    return offsets
        
f = open('imdb.csv', encoding='latin-1')
offsets = record_offsets(f)   

def data_generator(f, epochs, steps_per_epoch, batch_size, offsets):
    reader = csv.reader(f)
    for _ in range(epochs):
        random.shuffle(offsets)
        for batch_no in range(steps_per_epoch):
            x = []
            y = []
            for offset in offsets[batch_no * batch_size: batch_no * batch_size + 32]:
                f.seek(offset, 0)
                result = next(reader)
                x.append(result[0])
                y.append(1 if result[1] == 'positive' else 0)
            yield np.array(x), np.array(y)
           
df = pd.read_csv('imdb.csv')

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, TextVectorization

tv = TextVectorization(output_mode='count')
tv.adapt(df['review'])

model = Sequential(name='IMDB')

model.add(Input((1, ), dtype='string'))
model.add(tv)
model.add(Dense(128, activation='relu', name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(data_generator(f, EPOCHS, math.ceil(len(df) / BATCH_SIZE), BATCH_SIZE, offsets),
                 batch_size=BATCH_SIZE, epochs=EPOCHS)

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

"""
eval_result = model.evaluate(test_dataset_x, test_dataset_y, batch_size=32)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
 
# prediction

predict_df = pd.read_csv('predict-imdb.csv') 
predict_result = model.predict(predict_df)

for presult in predict_result[:, 0]:
    if (presult > 0.5):
        print('Positive')
    else:
        print('Negative')

"""