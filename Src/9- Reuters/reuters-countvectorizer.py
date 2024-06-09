import os
import re

training_dict = {}
test_dict = {}
cats = set()

with open('ReutersData/cats.txt') as f:
    for line in f:
        toklist = line.split()
        ttype, fname = toklist[0].split('/')
        if ttype == 'training':
            training_dict[fname] = toklist[1]
        else:
            if ttype == 'test':
                test_dict[fname] = toklist[1]
        cats.add(toklist[1])
    
vocab =  set()
training_texts = []
training_y = []
                
for fname in os.listdir('ReutersData/training'):
    with open('ReutersData/training/' + fname, encoding='latin-1') as f:
        text = f.read()
        training_texts.append(text)
        words = re.findall('[a-zA-Z0-9]+', text.lower())
        vocab.update(words)
        training_y.append(training_dict[fname])

test_texts = []
test_y = []
for fname in os.listdir('ReutersData/test'):
    with open('ReutersData/test/' + fname, encoding='latin-1') as f:
        text = f.read()
        test_texts.append(text)
        words = re.findall('[a-zA-Z0-9]+', text.lower())
        vocab.update(words)
        test_y.append(test_dict[fname])
            
vocab_dict = {word: index for index, word in enumerate(vocab)}

import pandas as pd

df_sw = pd.read_csv('ReutersData/stopwords', header=None)
sw = df_sw.iloc[:, 0].to_list()

sw +=  ['ain', 'aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'll', 'mon', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn']

# ---------------------    
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(dtype='uint8', stop_words=sw, encoding='latin-1')
cv.fit(training_texts + test_texts)

training_dataset_x = cv.transform(training_texts).todense()
test_dataset_x = cv.transform(test_texts).todense()

# ---------------------
import numpy as np
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, dtype='uint8')
ohe.fit(np.array(list(cats)).reshape(-1, 1))

training_dataset_y = ohe.transform(np.array(training_y).reshape(-1, 1))
test_dataset_y = ohe.transform(np.array(test_y).reshape(-1, 1))

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential(name='Reuters')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(128, activation='relu', name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(len(cats), activation='softmax', name='Output'))
model.summary()
            
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=10, validation_split=0.2)

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

eval_result = model.evaluate(test_dataset_x , test_dataset_y, batch_size=32)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

# prediction 
           
predict_texts = []
fnames = []
for fname in os.listdir('PredictData'):
    with open('PredictData/' + fname, encoding='latin-1') as f:
        text = f.read()
        predict_texts.append(text)
        fnames.append(fname)

# ---------------------        
predict_dataset_x = cv.transform(predict_texts).todense()       
# ---------------------

predict_result = model.predict(predict_dataset_x)
predict_indexes = np.argmax(predict_result, axis=1)

for index, pi in enumerate(predict_indexes):
    print(f'{fnames[index]} => {ohe.categories_[0][pi]}')
           
        
    

