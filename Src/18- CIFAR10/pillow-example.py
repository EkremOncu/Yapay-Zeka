from PIL import Image

image = Image.open("Predict-Pictures/frog-1.jpg")
resized_image = image.resize((32, 32))
resized_image.save('frog.bmp')

result = image.rotate(180)
result.show()

result = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
result.show()
image.show()