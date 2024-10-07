TEXT_SIZE = 250
WORD_VECT_SIZE = 64

import pandas as pd

df = pd.read_csv('IMDB Dataset.csv')

dataset_x = df['review']
dataset_y = (df['sentiment'] == 'positive').to_numpy(dtype='uint8')

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, TextVectorization, Conv1D, GlobalMaxPooling1D, Embedding, Dense, Reshape

tv = TextVectorization(output_sequence_length=TEXT_SIZE, output_mode='int')
tv.adapt(dataset_x)

model = Sequential(name='IMBD-WordEmbedding')
model.add(Input((1, ), dtype='string', name='Input'))

model.add(tv)

model.add(Embedding(tv.vocabulary_size(), WORD_VECT_SIZE, name='Embedding'))

model.add(Conv1D(128, 3, padding='same', name='Conv1D-1'))
model.add(GlobalMaxPooling1D(name='GlobalMaxPooling1D-1'))

model.add(Reshape((-1, ), name='Reshape'))

model.add(Dense(256, activation='relu', name='Hidden-1'))
model.add(Dense(256, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()