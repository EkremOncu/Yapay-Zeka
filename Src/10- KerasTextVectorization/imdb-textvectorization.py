"""
Burada yapmamız gereken şey Input katmanından sonra bu TextVectoriation katmanını 
modele eklemektir. Ancak Input katmanında girdilerin yazısal olduğunu belirtmek 
için Input fonksiyonunun dtype paramatresi 'string' girilmelidir. (Defult durumda 
Keras girdi katmanındaki değerlerin float türünden olduğunu kabul etmektedir.

output_mode parametresi "count" biçiminde girilirse bu durumda yazı, bizim 
istediğimiz gibi frekanslardan oluşan vektör biçimine dönüştürülecektir
"""

import pandas as pd

df = pd.read_csv('IMDB Dataset.csv')

dataset_x = df['review']  
dataset_y = (df['sentiment'] == 'positive').astype(dtype='uint8')

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(df['review']
                                                                        ,dataset_y, test_size=0.2)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, TextVectorization

tv = TextVectorization(output_mode='count')
tv.adapt(dataset_x)

model = Sequential(name='IMDB')

model.add(Input((1, ), dtype='string'))
model.add(tv)
model.add(Dense(128, activation='relu', name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])

hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, 
                 epochs=5, validation_split=0.2)

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Epoch - Binary Accuracy Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['binary_accuracy'])
plt.plot(hist.epoch, hist.history['val_binary_accuracy'])
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.show()

eval_result = model.evaluate(test_dataset_x, test_dataset_y, batch_size=32)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
 
# prediction

predict_df = pd.read_csv('predict-imdb.csv') 
predict_result = model.predict(predict_df)

for presult in predict_result[:, 0]:
    if (presult > 0.5):
        print('Positive')
    else:
        print('Negative')