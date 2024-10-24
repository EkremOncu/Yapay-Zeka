from tensorflow.keras.layers import Dropout
import numpy as np

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='float')

dropout_layer = Dropout(0.8)

result = dropout_layer(data, training=True)
print(result)

result = dropout_layer(data, training=True)
print(result)
