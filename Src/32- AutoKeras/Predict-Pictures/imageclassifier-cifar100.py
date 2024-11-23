import glob

EPOCHS = 5

from tensorflow.keras.datasets import cifar100

(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) = cifar100.load_data()

from tensorflow.keras.utils import to_categorical

ohe_training_dataset_y = to_categorical(training_dataset_y)
ohe_test_dataset_y = to_categorical(test_dataset_y)

class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]

import autokeras as ak

ic = ak.ImageClassifier(max_trials=2, metrics=['categorical_accuracy'],overwrite=True)

hist = ic.fit(training_dataset_x, training_dataset_y, epochs=1)

import matplotlib.pyplot as plt

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

eval_result = ic.evaluate(test_dataset_x, test_dataset_y)

for i in range(len(eval_result)):
    print(f'{eval_result[i]}')

# prediction

import numpy as np
import os

count = 0
hit_count = 0
for path in glob.glob('Predict-Pictures/*.*'):
    image = plt.imread(path)
    scaled_image = image / 255
    model_result = ic.predict(scaled_image.reshape(-1, 32, 32, 3), verbose=0)
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













