TEXT_SIZE = 250
WORD_VECT_SIZE = 64

import pandas as pd

df = pd.read_csv('IMDB Dataset.csv')

dataset_x = df['review']
dataset_y = (df['sentiment'] == 'positive').to_numpy(dtype='uint8')

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, TextVectorization, Conv1D, Embedding, Dense, Reshape

tv = TextVectorization(output_sequence_length=TEXT_SIZE, output_mode='int')
tv.adapt(dataset_x)

model = Sequential(name='IMBD-WordEmbedding')
model.add(Input((1, ), dtype='string', name='Input'))

model.add(tv)

model.add(Embedding(tv.vocabulary_size(), WORD_VECT_SIZE, name='Embedding'))

model.add(Conv1D(128, 3, padding='same', name='Conv1D'))

model.add(Reshape((-1, ), name='Reshape'))
model.add(Dense(256, activation='relu', name='Hidden-1'))
model.add(Dense(256, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

esc = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

hist = model.fit(training_dataset_x, training_dataset_y, epochs=10, batch_size=32, 
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
plt.title('Epoch - Binary Accuracy Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['binary_accuracy'])
plt.plot(hist.epoch, hist.history['val_binary_accuracy'])
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.show()

eval_result = model.evaluate(test_dataset_x, test_dataset_y)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

predict_df = pd.read_csv('predict-imdb.csv')

predict_result = model.predict(predict_df['review'])
for presult in predict_result[:, 0]:
    if (presult > 0.5):
        print('Positive')
    else:
        print('Negative')







