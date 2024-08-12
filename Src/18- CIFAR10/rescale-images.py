from PIL import Image
import glob

for path in glob.glob('Predict-Pictures/*.*'):
    image = Image.open(path)
    resized_image = image.resize((32, 32))
    image.close()
    """
    index = path.rindex('.')
    jpeg_path = path[:index] + '.jpg'
    resized_image.save(jpeg_path)
    """
    
    resized_image.save(path)
    
                      


