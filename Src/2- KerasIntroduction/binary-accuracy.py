from tensorflow.keras.metrics import binary_accuracy
import numpy as np

y = np.array([1, 0, 1, 1, 0])
y_hat = np.array([0.90, 0.7, 0.3, 0.9, 0.6])

result = binary_accuracy(y, y_hat)
print(result)