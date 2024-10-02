from tensorflow.keras.layers import Conv1D
import numpy as np

we_text = np.random.random((6, 8))

conv1d = Conv1D(1, 2)

result = conv1d(we_text.reshape((1, 6, 8))) # girdi 3 boyutlu olmalı 
print(result.shape)

