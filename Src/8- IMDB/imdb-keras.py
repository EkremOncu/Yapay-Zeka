from tensorflow.keras.datasets import imdb

(training_dataset_xx, training_dataset_y), (test_dataset_xx, test_dataset_y) = imdb.load_data()

vocab_dict = imdb.get_word_index()

import numpy as np

def vectorize(sequence, colsize):
    dataset_x = np.zeros((len(sequence), colsize), dtype='uint8')
    for index, vals in enumerate(sequence):
        dataset_x[index, vals] = 1
        
    return dataset_x


training_dataset_x = vectorize(training_dataset_xx, len(vocab_dict) + 3)
test_dataset_x = vectorize(test_dataset_xx, len(vocab_dict) + 3)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential(name='IMDB')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(128, activation='relu', name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=5, validation_split=0.2)

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

import numpy as np
import pandas as pd
import re
    
predict_df = pd.read_csv('predict-imdb.csv')

predict_lists = []
for text in predict_df['review']:
    index_list = []
    words = re.findall('[A-Za-z0-9]+', text.lower())
    for word in words:
        index_list.append(vocab_dict[word] + 3)
    predict_lists.append(index_list)
    
predict_dataset_x = vectorize(predict_lists, len(vocab_dict) + 3)

predict_result = model.predict(predict_dataset_x)

for presult in predict_result[:, 0]:
    if (presult > 0.5):
        print('Positive')
    else:
        print('Negative')

# dataset_x'teki birinci yorumun yazı haline getirilmesi

rev_vocab_dict = {index: word for word, index in vocab_dict.items()}

word_indices = np.argwhere(training_dataset_x[0] == 1).flatten()
words = [rev_vocab_dict[index - 3] for index in word_indices if index > 2]
text = ' '.join(words)
print(text)

    






    