import tensorflow
from tensorflow.keras.preprocessing import image_dataset_from_directory

dataset = image_dataset_from_directory('Images', label_mode='binary', image_size=(128, 128), 
                                       batch_size=1)

batch_data = dataset.take(-1)

import matplotlib.pyplot as plt

for x, y in batch_data:
    image = tensorflow.cast(x[0], 'uint8')
    plt.title(str(int(y)))
    plt.imshow(image)
    plt.show()
    


