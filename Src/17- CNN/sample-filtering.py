"""
Aşağıdaki örnekte evrişim işlemi yapan conv isimli bir fonksiyon yazılmıştır. Bu 
fonksiyonla "blur" ve "sobel" filtreleri denenmiştir. Görüntü işlemede blur filtresi 
resmi bulanıklaştırmakta sobel filtresi ise nesnelerin sınır çizgilerini belirgin 
hale getirmek kullanılmaktadır. Blur filtrelemesinde eğer resminizin pixel boyutları 
büyükse kernel matrisi daha büyük tutmalısınız.
"""

import numpy as np

def conv(image, filter):
    image_height = image.shape[0]
    image_width = image.shape[1]
    filter_height = filter.shape[0]
    filter_width = filter.shape[1]

    conv_height = image_height - filter_height + 1
    conv_width = image_width - filter_width + 1

    conv_image = np.zeros((conv_height, conv_width), dtype=np.uint8)

    for row in range(conv_height):
        for col in range(conv_width):
            dotp = 0
            for i in range(filter_height):
                for j in range(filter_width):
                    dotp += image[row + i, col + j] * filter[i, j]
            conv_image[row, col] = np.clip(dotp, 0, 255)

    return conv_image

import matplotlib.pyplot as plt

image = plt.imread('AbbeyRoad.jpg')
gray_scaled_image = np.average(image, axis=2, weights=[0.3, 0.59, 0.11])
plt.imshow(gray_scaled_image, cmap='gray')
plt.show()

blur_kernel = np.full((30,30), 1 / 100)
convoluted_image = conv(gray_scaled_image, blur_kernel)
plt.imshow(convoluted_image, cmap='gray')
plt.show()

sobel_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
convoluted_image = conv(gray_scaled_image, sobel_kernel)
plt.imshow(convoluted_image, cmap='gray')
plt.show()

