import pandas as pd


df = pd.read_csv('titanic-dataset.csv')


dataset_x = df.drop(columns=['Survived'], axis=1)
dataset_y = df['Survived'].to_numpy()


from sklearn.model_selection import train_test_split


training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, 
                                                                                          dataset_y, test_size=0.2)


import autokeras as ak


sdc = ak.StructuredDataClassifier(max_trials=10, overwrite=True)
hist = sdc.fit(training_dataset_x, training_dataset_y, epochs=100, validation_split=0.2)


import matplotlib.pyplot as plt


plt.figure(figsize=(15, 5))
plt.title('Epoch-Loss Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')


plt.plot(hist.epoch, hist.history['loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()


plt.figure(figsize=(15, 5))
plt.title('Epoch-Binary Accuracy Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')


plt.plot(hist.epoch, hist.history['accuracy'])
plt.legend(['Binary Accuracy', 'Validation Binary Accuracy'])
plt.show()


model = sdc.export_model()
model.summary()
model.save('text-classifier-best-model.tf', save_format='tf')


eval_result = sdc.evaluate(test_dataset_x, test_dataset_y)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
