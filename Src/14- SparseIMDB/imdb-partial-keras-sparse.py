import pandas as pd

EPOCHS = 5
BATCH_SIZE = 32

df = pd.read_csv('IMDB Dataset.csv').iloc[:1000, :]

from sklearn.feature_extraction.text  import CountVectorizer

cv = CountVectorizer(dtype='uint8')

dataset_x = cv.fit_transform(df['review'])
dataset_y = (df['sentiment'] == 'positive').to_numpy(dtype='uint8') 

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, 
                                                                        dataset_y, test_size=0.2)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential(name='IMDB')

model.add(Input((training_dataset_x.shape[1],), sparse=True)) # !!! sparse= True
model.add(Dense(128, activation='relu', name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=BATCH_SIZE, 
                 epochs=EPOCHS, validation_split=0.2)
    
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['loss'])
plt.legend(['Loss'])
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

df_predict = pd.read_csv('predict-imdb.csv')
predict_dataset_x = cv.transform(df_predict['review'])

predict_result = model.predict(predict_dataset_x)

for presult in predict_result[:, 0]:
    if (presult > 0.5):
        print('Positive')
    else:
        print('Negative')





