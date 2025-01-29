from tensorflow.keras.datasets import mnist

(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) = mnist.load_data()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.title(str(training_dataset_y[i]), fontsize=14)
    plt.imshow(training_dataset_x[i], cmap='gray')
    
plt.show()

from tensorflow.keras.utils import to_categorical

ohe_training_dataset_y = to_categorical(training_dataset_y)
ohe_test_dataset_y = to_categorical(test_dataset_y)

scaled_training_dataset_x = training_dataset_x.reshape(-1, 784) / 255
scaled_test_dataset_x = test_dataset_x.reshape(-1, 784) / 255

import numpy as np
from sklearn.decomposition import PCA

total_ratios = []
for i in range(1, 300):
    pca = PCA(i)
    pca.fit(scaled_training_dataset_x)
    total_ratio = np.sum(pca.explained_variance_ratio_)
    total_ratios.append(total_ratio)
    print(i, end=' ')
 
import pickle

with open('mnist.pickle', 'wb') as f:
    pickle.dump(total_ratios, f)
              
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.title('Optimal number of Featured')
plt.plot(range(1, 300), total_ratios, color='red')
plt.plot(range(1, 300), total_ratios, 'bo', color='blue')
plt.legend(['Total explained variance ratio'])
plt.xlabel('Nuber of Features')
plt.ylabel('Ratio')
plt.xticks(range(1, 300, 10))

plt.show()