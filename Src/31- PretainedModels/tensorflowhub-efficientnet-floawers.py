from tensorflow.keras.preprocessing import image_dataset_from_directory

IMAGE_SIZE = (250, 250)

training_dataset = image_dataset_from_directory(r'flower_photos', label_mode='categorical',
      subset='training', seed=123, validation_split=0.2, image_size=IMAGE_SIZE,  batch_size=1)

valdation_dataset = image_dataset_from_directory(r'flower_photos', label_mode='categorical',
      subset='validation', seed=123, validation_split=0.2, image_size=IMAGE_SIZE,  batch_size=1)


from tensorflow_hub import KerasLayer

keras_layer = KerasLayer('https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2')


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Rescaling, Dense

model = Sequential(name='Tensorflow-Hub-EfficentNet')

model.add(Input(IMAGE_SIZE + (3, ), name='Input'))
model.add(Rescaling(1. / 255, name='Rescaling'))
model.add(keras_layer)
model.add(Dense(5, activation='softmax', name='Output'))












