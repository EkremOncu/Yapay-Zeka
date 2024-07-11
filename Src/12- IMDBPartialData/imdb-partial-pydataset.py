import math
import random
import csv
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import PyDataset

EPOCHS = 5
BATCH_SIZE = 32
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2

def record_offsets(f, skiprows=1):
    offsets = []
    for i in range(skiprows):
        f.readline()
    offsets.append(f.tell())
    while f.readline() != '':
        offsets.append(f.tell())
    offsets.pop()
        
    return offsets
        
f = open('imdb.csv', encoding='latin-1')
offsets = record_offsets(f)   

random.shuffle(offsets)

test_split_index = int(len(offsets) * (1 - TEST_RATIO))
validation_split_index = int(test_split_index * (1 - VALIDATION_RATIO))

training_offsets = offsets[:validation_split_index]
validation_offsets = offsets[validation_split_index:test_split_index]
test_offsets = offsets[test_split_index:]

training_steps = math.ceil(len(training_offsets) / BATCH_SIZE)
validation_steps = math.ceil(len(validation_offsets)/BATCH_SIZE)
test_steps = math.ceil(len(test_offsets)/BATCH_SIZE)

class DataGenerator(PyDataset):
    def __init__(self, f, steps, batch_size, offsets, *, shuffle=False, predict=False):
        super().__init__()
        self.f = f
        self.steps = steps
        self.batch_size = batch_size
        self.offsets = offsets
        self.shuffle = shuffle
        self.predict = predict
        self.reader = csv.reader(f)
        
    def __len__(self):
        return self.steps
    
    def __getitem__(self, batch_no):        
        x = []
        if not self.predict:
            y = []
        for offset in self.offsets[batch_no * self.batch_size: batch_no * self.batch_size + self.batch_size]:
            f.seek(offset, 0)
            result = next(self.reader)
            x.append(result[0])
            if not self.predict:
                y.append(1 if result[1] == 'positive' else 0)
        if not self.predict:
            return tf.convert_to_tensor(x), tf.convert_to_tensor(y)
        return tf.convert_to_tensor(x),
    
    def on_pech_end(self):
        random.shuffle(self.offsets)  

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

hist = model.fit(DataGenerator(f, training_steps, BATCH_SIZE, training_offsets, shuffle=True), 
                epochs=EPOCHS, validation_data = DataGenerator(f, validation_steps, 
                BATCH_SIZE, validation_offsets), verbose=0) 
                 

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['loss'])
plt.legend(['Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Epoch - Binary Accuracy Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['binary_accuracy'])
plt.plot(hist.epoch, hist.history['val_binary_accuracy'])
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.show()


eval_result = model.evaluate(DataGenerator(f, test_steps, BATCH_SIZE, test_offsets))
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

# prediction

predict_f = open('predict-imdb.csv')
predict_offsets = record_offsets(predict_f)
predict_steps = int(math.ceil(len(predict_offsets) / BATCH_SIZE))

predict_result = model.predict(DataGenerator(f, predict_steps, BATCH_SIZE, predict_offsets, predict=True))

for presult in predict_result[:, 0]:
    if (presult > 0.5):
        print('Positive')
    else:
        print('Negative')





