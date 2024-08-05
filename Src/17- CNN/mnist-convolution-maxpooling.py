import pandas as pd

training_df = pd.read_csv('mnist_train.csv')
test_df = pd.read_csv('mnist_test.csv')

training_dataset_x = training_df.iloc[:, 1:].to_numpy(dtype='uint8')
training_dataset_y = training_df.iloc[:, 0].to_numpy(dtype='uint8')

test_dataset_x = test_df.iloc[:, 1:].to_numpy(dtype='uint8')
test_dataset_y = test_df.iloc[:, 0]

# one hot encoding for y data

from tensorflow.keras.utils import to_categorical

ohe_training_dataset_y = to_categorical(training_dataset_y)
ohe_test_dataset_y = to_categorical(test_dataset_y)


# reshape as 28x28x1

training_dataset_x = training_dataset_x.reshape(-1, 28, 28, 1)
test_dataset_x = test_dataset_x.reshape(-1, 28, 28, 1)


import matplotlib.pyplot as plt

plt.figure(figsize=(5, 13))
for i in range(50):
    plt.subplot(10, 5, i + 1)
    plt.title(str(training_dataset_y[i]), fontsize=12, fontweight='bold')
    picture = training_dataset_x[i]
    plt.imshow(picture, cmap='gray')
plt.show()


# minmax scaling

scaled_training_dataset_x = training_dataset_x / 255
scaled_test_dataset_x = test_dataset_x / 255

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential(name='MNIST')
model.add(Input((28, 28, 1), name='Input'))
model.add(Conv2D(32, (3, 3), activation='relu', name='Conv2D-1'))
model.add(MaxPooling2D(name='MaxPooling2D-1'))
model.add(Conv2D(64, (3, 3), activation='relu', name='Conv2D-2'))
model.add(MaxPooling2D(name='MaxPooling2D-2'))
model.add(Flatten(name='Flatten'))    
model.add(Dense(128, activation='relu', name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(10, activation='softmax', name='Output'))
model.summary()

model.compile('rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

hist = model.fit(scaled_training_dataset_x, ohe_training_dataset_y, batch_size=32, 
                 epochs=20, validation_split=0.2)

plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', fontsize=14, pad=10)
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Categorcal Accuracy - Validation Categorical Accuracy', fontsize=14, pad=10)
plt.plot(hist.epoch, hist.history['categorical_accuracy'])
plt.plot(hist.epoch, hist.history['val_categorical_accuracy'])
plt.legend(['Categorical Accuracy', 'Validation Categorical Accuracy'])
plt.show()

eval_result = model.evaluate(scaled_test_dataset_x , ohe_test_dataset_y, batch_size=32)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

# prediction

import numpy as np
import os
import glob

for path in glob.glob('Predict-Pictures/*.bmp'):
    image = plt.imread(path)
    gray_scaled_image = np.average(image, axis=2, weights=[0.3, 0.59, 0.11])
    gray_scaled_image = gray_scaled_image.reshape(1, 28, 28, 1)
    gray_scaled_image /= 255
    
    model_result = model.predict(gray_scaled_image, verbose=0)
    predict_result = np.argmax(model_result)
    print(f'Real Number: {os.path.basename(path)[0]}, Prdicted Result: {predict_result}, Path: {path}')
    
    



















    


