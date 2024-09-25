TEXT_SIZE = 250
WORD_VECT_SIZE = 2

import pandas as pd

df = pd.read_csv('IMDB Dataset.csv')

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

cv.fit(df['review'])

import re

text_vectors = [[cv.vocabulary_[word] + 1  for word in re.findall(r'(?u)\b\w\w+\b', text.lower())] for text in df['review']]

from tensorflow.keras.utils import pad_sequences

dataset_x = pad_sequences(text_vectors, TEXT_SIZE, dtype='float32')
dataset_y = (df['sentiment'] == 'positive').to_numpy(dtype='uint8')


from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y)


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense

model = Sequential(name='IMBD-WordEmbedding')

model.add(Input((TEXT_SIZE, ), name='Input'))

model.add(Embedding(len(cv.vocabulary_) + 1, WORD_VECT_SIZE, name='Embedding'))


model.add(Flatten(name='Flatten'))
model.add(Dense(256, activation='relu', name='Hidden-1'))
model.add(Dense(256, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

esc = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

hist = model.fit(training_dataset_x, training_dataset_y, epochs=100, batch_size=32, 
                 validation_split=0.2, callbacks=[esc])

# model test 

text = 'good bad awful perfect extraordinary well impressive average magnificent disgusting best poor ok terrible cool worst fine moderate satisfactory valuable super inferior'

words = re.findall(r'(?u)\b\w\w+\b', text)
word_indexes = [cv.vocabulary_[word] + 1 for word in words]
    
input_vect = pad_sequences([word_indexes], TEXT_SIZE, padding='post', truncating='post')
output_vect = model.layers[0](input_vect)

word_values = output_vect[0, :len(word_indexes), :]


import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))

plt.title('Word Embedding Visual Representation')

plt.scatter(word_values[:, 0], word_values[:, 1])

for i, name in enumerate(words):
    plt.annotate(name, (word_values[i, 0] + 0.005, word_values[i, 1] + 0.005))

plt.show()






