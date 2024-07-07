import math
import random
import csv
import pandas as pd
import tensorflow as tf

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

def training_validation_test_generator(f, epochs, steps, batch_size, offsets, shuffle=False):
    reader = csv.reader(f)
    for _ in range(epochs):
        if shuffle:
            random.shuffle(offsets)
        for batch_no in range(steps):
            x = []
            y = []
            for offset in offsets[batch_no * batch_size: batch_no * batch_size + batch_size]:
                f.seek(offset, 0)
                result = next(reader)
                x.append(result[0])
                y.append(1 if result[1] == 'positive' else 0)
            yield tf.convert_to_tensor(x), tf.convert_to_tensor(y)
                
def predict_generator(f, steps, batch_size, offsets):
    reader = csv.reader(f)
    for batch_no in range(steps):
        x = []
        for offset in offsets[batch_no * batch_size:batch_no * batch_size + batch_size]:
            f.seek(offset, 0)
            result = next(reader)
            x.append(result[0])
        yield tf.convert_to_tensor(x), 
                  
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
hist = model.fit(training_validation_test_generator(f, 
                 EPOCHS, training_steps, BATCH_SIZE, training_offsets, True), 
                 batch_size=BATCH_SIZE, 
                 steps_per_epoch=training_steps, 
                 epochs=EPOCHS, 
                 validation_data=training_validation_test_generator(f, EPOCHS, validation_steps, BATCH_SIZE, validation_offsets)
                 ,validation_steps=validation_steps)

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

eval_result = model.evaluate(training_validation_test_generator(f, 1, test_steps, 
                                BATCH_SIZE, test_offsets), steps=test_steps)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

# prediction

predict_f = open('predict-imdb.csv')
predict_offsets = record_offsets(predict_f)
predict_steps = int(math.ceil(len(predict_offsets) / BATCH_SIZE))

predict_result = model.predict(predict_generator(predict_f, predict_steps, BATCH_SIZE, 
                                    predict_offsets), steps=predict_steps)

for presult in predict_result[:, 0]:
    if (presult > 0.5):
        print('Positive')
    else:
        print('Negative')


    
















