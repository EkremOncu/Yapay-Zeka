import pandas as pd

EPOCHS = 200

df = pd.read_csv('housing.csv', delimiter=r'\s+', header=None)

highway_class = df.iloc[:, 8].to_numpy()

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)
ohe_highway = ohe.fit_transform(highway_class.reshape(-1, 1))

dataset_y = df.iloc[:, -1].to_numpy()
df.drop([8, 13], axis=1, inplace=True)
dataset_x = pd.concat([df, pd.DataFrame(ohe_highway)], axis=1).to_numpy()

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.1)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(training_dataset_x)
scaled_training_dataset_x = ss.transform(training_dataset_x)
scaled_test_dataset_x = ss.transform(test_dataset_x)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

model = Sequential(name='Boston-Housing-Prices')
model.add(Input((training_dataset_x.shape[1], ), name='Input'))
model.add(Dense(64, activation='relu', name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='linear', name='Output'))
model.summary()

model.compile('rmsprop', loss='mse', metrics=['mae'])

mcp = ModelCheckpoint('Boston-Housing-{epoch:03d}.keras', monitor='val_loss', 
                      save_best_only=True)

esc = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, 
                    verbose=1, mode='min')

hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size=32, 
                 epochs=EPOCHS, validation_split=0.2, callbacks=[mcp, esc])


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

eval_result = model.evaluate(scaled_test_dataset_x , test_dataset_y, batch_size=32)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')


# prediction

predict_df = pd.read_csv('predict-boston-housing-prices.csv', delimiter=r'\s+', header=None)

highway_class = predict_df.iloc[:, 8].to_numpy()
ohe_highway = ohe.transform(highway_class.reshape(-1, 1))
predict_df.drop(8, axis=1, inplace=True)
predict_dataset_x = pd.concat([predict_df, pd.DataFrame(ohe_highway)], axis=1).to_numpy()
scaled_predict_dataset_x = ss.transform(predict_dataset_x )
predict_result = model.predict(scaled_predict_dataset_x)

for val in predict_result[:, 0]:
    print(val)
    


















