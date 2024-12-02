from tensorflow.keras.datasets import mnist

(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) = mnist.load_data()


import autokeras as ak

inp = ak.ImageInput()
x = ak.ConvBlock()(inp)
x = ak.DenseBlock(num_layers=2)(x)
out = ak.ClassificationHead()(x)
auto_model = ak.AutoModel(inputs=inp, outputs=out, max_trials=1, overwrite=True)

hist = auto_model.fit(training_dataset_x, training_dataset_y, validation_split=0.2, epochs=5)

exported_model = auto_model.export_model()
exported_model.summary()


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

# evaluation

eval_result = auto_model.evaluate(test_dataset_x , test_dataset_y, batch_size=32)
for i in range(len(eval_result)):
    print(f'{exported_model.metrics_names[i]}: {eval_result[i]}')
    
    
    