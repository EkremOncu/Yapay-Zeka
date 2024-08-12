import glob

EPOCHS = 5

from tensorflow.keras.datasets import cifar10

(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) = cifar10.load_data()

from tensorflow.keras.utils import to_categorical

ohe_training_dataset_y = to_categorical(training_dataset_y)
ohe_test_dataset_y = to_categorical(test_dataset_y)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 
               'horse', 'ship', 'truck']
  
import matplotlib.pyplot as plt
     
plt.figure(figsize=(4, 20))
for i in range(30):
    plt.subplot(10, 3, i + 1)
    plt.title(class_names[training_dataset_y[i, 0]], pad=10)    
    plt.imshow(training_dataset_x[i])
plt.show()

scaled_training_dataset_x = training_dataset_x / 255
scaled_test_dataset_x = test_dataset_x / 255

from tensorflow.keras.utils import to_categorical

ohe_training_dataset_y = to_categorical(training_dataset_y)
ohe_test_dataset_y = to_categorical(test_dataset_y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential(name='CIFAR10')
model.add(Input((32, 32, 3), name='Input'))
model.add(Conv2D(32, (3, 3), activation='relu', name='Conv2D-1'))
model.add(MaxPooling2D(name='MaxPooling2D-1'))
model.add(Conv2D(64, (3, 3), activation='relu', name='Conv2D-2'))
model.add(MaxPooling2D(name='MaxPooling2D-2'))
model.add(Conv2D(128, (3, 3), activation='relu', name='Conv2D-3'))
model.add(MaxPooling2D(name='MaxPooling2D-3'))
model.add(Flatten(name='Flatten'))    
model.add(Dense(256, activation='relu', name='Hidden-1'))
model.add(Dense(256, activation='relu', name='Hidden-2'))
model.add(Dense(10, activation='softmax', name='Output'))
model.summary()

model.compile('rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
hist = model.fit(scaled_training_dataset_x, ohe_training_dataset_y, batch_size=32,
                 epochs=EPOCHS, validation_split=0.2)

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

count = 0
hit_count = 0
for path in glob.glob('Predict-Pictures/*.*'):
    image = plt.imread(path)
    scaled_image = image / 255
    model_result = model.predict(scaled_image.reshape(-1, 32, 32, 3), verbose=0)
    predict_result = np.argmax(model_result)
    fname = os.path.basename(path)
    real_class = fname[:fname.index('-')]
    predict_class = class_names[predict_result]
    print(f'Real class: {real_class}, Predicted Class: {predict_class}, Path: {path}')
    
    if real_class == predict_class:
        hit_count += 1
    count += 1
    
print('-' * 20)
print(f'Prediction accuracy: {hit_count / count}')
    
    





