import matplotlib.pyplot as plt


picture = plt.imread("C:\\Users\\Lenovo\\Desktop\\GitHub\\YapayZeka\\Src\\20- ImageAugmentation\\Sample-Pictures\AbbeyRoad.jpg")
plt.figure(figsize=(9, 16))
plt.imshow(picture);
plt.show()

from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, Rescaling, RandomCrop, Resizing, RandomContrast

rf = RandomFlip('horizontal')
result = rf(picture).numpy()
plt.figure(figsize=(9, 16))
plt.imshow(result.astype('uint8'));
plt.show()

rr = RandomRotation(0.1)
result = rr(picture).numpy()
plt.figure(figsize=(9, 16))
plt.imshow(result.astype('uint8'));
plt.show()

rz = RandomZoom(0.3)
result = rz(picture).numpy()
plt.figure(figsize=(9, 16))
plt.imshow(result.astype('uint8'));
plt.show()

rs = Rescaling(0.50)
result = rs(picture).numpy()
plt.figure(figsize=(9, 16))
plt.imshow(result.astype('uint8'));
plt.show()

rc = RandomCrop(500, 500)
result1 = rc(picture).numpy() # result1.shape --> (500, 500, 3)
plt.figure(figsize=(9, 16))
rs = Resizing(1000, 1000)
result = rs(result1).numpy()  # result.shape --> (1000, 1000, 3)
plt.figure(figsize=(9, 16))
plt.imshow(result.astype('uint8'));
plt.show()

rc = RandomContrast(0.2)
result = rc(picture).numpy()
plt.figure(figsize=(9, 16))
plt.imshow(result.astype('uint8'));
plt.show()









