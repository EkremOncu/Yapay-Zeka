from tensorflow.keras.preprocessing import image_dataset_from_directory

IMAGE_SIZE = (250, 250)

training_dataset = image_dataset_from_directory(r'flower_photos', label_mode='categorical',
      subset='training', seed=123, validation_split=0.2, image_size=IMAGE_SIZE,  batch_size=1)

validation_dataset = image_dataset_from_directory(r'flower_photos', label_mode='categorical',
      subset='validation', seed=123, validation_split=0.2, image_size=IMAGE_SIZE,  batch_size=1)


from tensorflow_hub import KerasLayer


keras_layer = KerasLayer('https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2')


from tf_keras import Sequential
from tf_keras.layers import Input, Rescaling, Dense


model = Sequential(name='Tensorflow-Hub-EfficentNet')
model.add(Input(IMAGE_SIZE + (3, ), name='Input'))
model.add(Rescaling(1. / 255, name='Rescaling'))
model.add(keras_layer)
model.add(Dense(5, activation='softmax', name='Output'))


model.compile('rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
hist = model.fit(training_dataset, validation_data=validation_dataset, epochs=3)


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











