import matplotlib.pyplot as plt
import numpy as np

image_data = plt.imread('AbbeyRoad.jpg')

plt.imshow(image_data)
plt.show()
print(image_data.shape)

result_image = np.average(image_data, axis=2, weights=[0.3, 0.59, 0.11])
plt.imshow(result_image, cmap='gray')
plt.show()
print(result_image.shape)


