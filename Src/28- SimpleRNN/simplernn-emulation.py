import numpy as np

TIMESPEC = 100                      # ardışıl uygulanacak eleman sayısı
INPUT_FEATURES = 32                 # girdi katmanındaki nöron sayısı
RECURRENT_LAYER_SIZE = 64           # geri besleme katmanındaki nöron sayısı

def activation(x):
    return np.maximum(0, x)

inputs = np.random.random((TIMESPEC, INPUT_FEATURES))
W = np.random.random((RECURRENT_LAYER_SIZE, INPUT_FEATURES))
U = np.random.random((RECURRENT_LAYER_SIZE, RECURRENT_LAYER_SIZE))
b = np.random.random((RECURRENT_LAYER_SIZE,))

outputs = []

output_t = np.random.random((RECURRENT_LAYER_SIZE,))
for input_t in inputs:
    output_t = activation(np.dot(W, input_t) + np.dot(U, output_t) + b)
    outputs.append(output_t)

total_outputs = np.concatenate(outputs, axis=0)

print(total_outputs)