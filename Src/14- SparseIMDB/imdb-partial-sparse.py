import math
import pandas as pd
from tensorflow.keras.utils import PyDataset
import tensorflow as tf
from sklearn.utils import shuffle

EPOCHS = 5
BATCH_SIZE = 32

df = pd.read_csv('IMDB Dataset.csv')

from sklearn.feature_extraction.text  import CountVectorizer

cv = CountVectorizer(dtype='uint8')

dataset_x = cv.fit_transform(df['review'])
dataset_y = (df['sentiment'] == 'positive').to_numpy(dtype='uint8') 

from sklearn.model_selection import train_test_split

temp_dataset_x, test_dataset_x, temp_dataset_y, test_dataset_y = train_test_split(dataset_x, 
                                                                dataset_y, test_size=0.2)

training_dataset_x, validation_dataset_x, training_dataset_y, validation_dataset_y = train_test_split(temp_dataset_x, 
                                                                                temp_dataset_y, test_size=0.2)

training_steps = math.ceil(training_dataset_x.shape[0] / BATCH_SIZE)
validation_steps = math.ceil(validation_dataset_x.shape[0] / BATCH_SIZE)
test_steps = math.ceil(test_dataset_x.shape[0] / BATCH_SIZE)

class DataGenerator(PyDataset):
    def __init__(self, sparse_x, y, steps, batch_size, predict=False):
        super().__init__()
        self.sparse_x = sparse_x
        self.y = y
        self.steps = steps
        self.batch_size = batch_size
        self.predict = predict
        
    def __len__(self):
        return self.steps
    
    def __getitem__(self, batch_no):      
        x = self.sparse_x[batch_no * self.batch_size: batch_no * self.batch_size + self.batch_size]
        if not self.predict:
            y = self.y[batch_no * self.batch_size: batch_no * self.batch_size + self.batch_size]
            return tf.convert_to_tensor(x.todense()), tf.convert_to_tensor(y)
        else:
            return tf.convert_to_tensor(x.todense()),
       
    def on_epoch_end(self):
        if not self.predict:
            self.sparse_x, self.y = shuffle(self.sparse_x, self.y)
          
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential(name='IMDB')

model.add(Input((training_dataset_x.shape[1], )))
model.add(Dense(128, activation='relu', name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(DataGenerator(training_dataset_x, training_dataset_y, training_steps, BATCH_SIZE), 
                 validation_data = DataGenerator(training_dataset_x, training_dataset_y, validation_steps, BATCH_SIZE), 
                 epochs=EPOCHS)
    
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

eval_result = model.evaluate(DataGenerator(test_dataset_x, test_dataset_y, test_steps, BATCH_SIZE))

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

df_predict = pd.read_csv('predict-imdb.csv')
predict_dataset_x = cv.transform(df_predict['review'])
predict_steps = math.ceil(predict_dataset_x.shape[0] / BATCH_SIZE)

predict_result = model.predict(DataGenerator(predict_dataset_x, None, predict_steps, BATCH_SIZE, True))

for presult in predict_result[:, 0]:
    if (presult > 0.5):
        print('Positive')
    else:
        print('Negative')





