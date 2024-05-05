import numpy as np

dataset = np.array([[1, 2, 3], [4, 5, 6], [3, 2, 7], [5, 9, 5]])
print(dataset)
print('--------')

from tensorflow.keras.layers import Normalization

norm_layer = Normalization()
norm_layer.adapt(dataset)

print(norm_layer.mean)
print('--------')
print(norm_layer.variance)

