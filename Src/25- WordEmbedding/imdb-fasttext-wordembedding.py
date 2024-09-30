TEXT_SIZE = 250
WORD_VECT_SIZE = 300

# ----------------------------------------------
FASTTEXT_WORD_EMBEDDING_FILE = 'cc.en.300.vec'

import numpy as np

we_dict = {}
with  open(FASTTEXT_WORD_EMBEDDING_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        tokens = line.rstrip().split(' ')
        we_dict[tokens[0]] = np.array([float(vecdata) for vecdata in tokens[1:]], dtype='float32')


# ----------------------------------------------

import pandas as pd

df = pd.read_csv('IMDB Dataset.csv')

dataset_x = df['review']
dataset_y = (df['sentiment'] == 'positive').to_numpy(dtype='uint8')


from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y)


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, TextVectorization, Embedding, Dense, GlobalAveragePooling1D

tv = TextVectorization(output_sequence_length=TEXT_SIZE, output_mode='int')
tv.adapt(dataset_x)
vocab_list = tv.get_vocabulary()


# ----------------------------------------------
pretrained_matrix = np.zeros((len(vocab_list), WORD_VECT_SIZE), dtype='float32')

for index, word in enumerate(vocab_list):
    vect = we_dict.get(word)
    if vect is None:
        vect = np.zeros(WORD_VECT_SIZE)
    pretrained_matrix[index] = vect

# ----------------------------------------------
        
model = Sequential(name='IMBD-WordEmbedding')
model.add(Input((1, ), dtype='string', name='Input'))
model.add(tv)

model.add(Embedding(len(vocab_list), WORD_VECT_SIZE, weights=[pretrained_matrix], name='Embedding'))

model.add(GlobalAveragePooling1D())

model.add(Dense(256, activation='relu', name='Hidden-1'))
model.add(Dense(256, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

esc = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

hist = model.fit(training_dataset_x, training_dataset_y, epochs=5, batch_size=32, 
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








