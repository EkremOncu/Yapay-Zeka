from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Reshape, Dense

model = Sequential(name='SimpleRNN-Test')
model.add(Input((100, 10),  name='Input'))

model.add(SimpleRNN(128,  return_sequences=True, name='SimpleRNN'))

model.add(Reshape((-1, ), name='Reshape'))

model.add(Dense(256, activation='relu', name='Hidden-1'))
model.add(Dense(256, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='linear', name='Output'))
model.summary()

