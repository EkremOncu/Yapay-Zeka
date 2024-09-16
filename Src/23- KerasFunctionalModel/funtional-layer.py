import numpy as np
from tensorflow.keras.layers import Input, Dense

data = np.random.random((32, 8))

inp = Input((8, ))
d = Dense(16, activation='relu', name='Dense')

result = d(inp)

result(data)


