import pandas as pd

dataset = pd.read_csv('dataset.csv')
dataset_x = dataset.iloc[:, :-1]
dataset_y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.1)

training_dataset_x1 = training_dataset_x.iloc[:, :-1].to_numpy(dtype='float32')
training_dataset_x2 = training_dataset_x.iloc[:, -1]

test_dataset_x1 = test_dataset_x.iloc[:, :-1].to_numpy(dtype='float32')
test_dataset_x2 = test_dataset_x.iloc[:, -1]

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(training_dataset_x1)
scaled_training_dataset_x1 = ss.transform(training_dataset_x1)
scaled_test_dataset_x1 = ss.transform(test_dataset_x1)


from tensorflow.keras import Model
from tensorflow.keras.layers import Input, TextVectorization, Dense, Concatenate

inp1 = Input(shape=(training_dataset_x1.shape[1], ), name='Numeric-Input')
inp2 = Input(shape=(1, ), dtype='string', name='Text-Input')

tv = TextVectorization(output_mode='count')
tv.adapt(training_dataset_x2)

result = tv(inp2)
result = Concatenate()([inp1, result])

result = Dense(128, activation='relu', name='Hidden-1')(result)
result = Dense(128, activation='relu', name='Hidden-2')(result)
out = Dense(1, activation='linear', name='Output')(result)

model = Model(inputs=[inp1, inp2], outputs=[out], name='MixedRandomModel')
model.summary()

model.compile('rmsprop', loss='mse', metrics=['mae'])

hist = model.fit([scaled_training_dataset_x1, training_dataset_x2], training_dataset_y, 
                 batch_size=32, epochs=100, validation_split=0.2)


import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', fontsize=14, pad=10)
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Mean Absolute Error - Validation Mean Absolute Error Graph', fontsize=14, pad=10)
plt.plot(hist.epoch, hist.history['mae'])
plt.plot(hist.epoch, hist.history['val_mae'])
plt.legend(['Mean Absolute Error', 'Validation Mean Absolute Error'])
plt.show()

eval_result = model.evaluate([scaled_test_dataset_x1, test_dataset_x2], test_dataset_y)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

predict_dataset = pd.read_csv('predict.csv')

predict_dataset_x1 = predict_dataset.iloc[:, :-1]
predict_dataset_x2 = predict_dataset.iloc[:, -1]

scaled_predict_dataset_x1 = ss.transform(predict_dataset_x1)
predict_result = model.predict([scaled_predict_dataset_x1, predict_dataset_x2])

for val in predict_result[:, 0]:
    print(val)
    



