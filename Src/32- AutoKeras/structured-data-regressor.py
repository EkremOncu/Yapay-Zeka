import pandas as pd

df = pd.read_csv('housing.csv', delimiter=r'\s+', header=None)

dataset_x = df.iloc[:, :-1].to_numpy(dtype='float32')
dataset_y = df.iloc[:, -1].to_numpy(dtype='float32')

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, 
                                                                                          test_size=0.2)



import autokeras as ak

sdr = ak.StructuredDataRegressor(max_trials=10, overwrite=True, metrics=['mae'])



from tensorflow.keras.callbacks import EarlyStopping

esc = EarlyStopping(patience=5, restore_best_weights=True)

hist = sdr.fit(training_dataset_x, training_dataset_y, epochs=100, callbacks=[esc])



import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.title('Epoch-Loss Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.plot(hist.epoch, hist.history['loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

model = sdr.export_model()

eval_result = sdr.evaluate(test_dataset_x, test_dataset_y)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
    
import numpy as np

predict_data = np.array([[0.11747, 12.50, 7.870, 0, 0.5240, 6.0090, 82.90, 6.2267, 5, 311.0, 15.20, 396.90, 13.27]])

predict_result = model.predict(predict_data)

for val in predict_result[:, 0]:
    print(val)
    

       