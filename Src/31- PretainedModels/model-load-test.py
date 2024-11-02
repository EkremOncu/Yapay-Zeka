from tensorflow.keras.models import load_model

model = load_model('ResNet50.keras')

model.summary()