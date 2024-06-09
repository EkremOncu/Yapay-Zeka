from tensorflow.keras.datasets import reuters

(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) = reuters.load_data()

vocab_dict = reuters.get_word_index()

import numpy as np

def vectorize(sequence, colsize):
    dataset_x = np.zeros((len(sequence), colsize), dtype='uint8')
    for index, vals in enumerate(sequence):
        dataset_x[index, vals] = 1
        
    return dataset_x

training_dataset_x = vectorize(training_dataset_x, len(vocab_dict) + 3)
test_dataset_x = vectorize(test_dataset_x, len(vocab_dict) + 3)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, dtype='uint8')
ohe.fit(np.concatenate([training_dataset_y, test_dataset_y]).reshape(-1, 1))

training_dataset_y = ohe.transform(training_dataset_y.reshape(-1, 1))
test_dataset_y = ohe.transform(test_dataset_y.reshape(-1, 1))

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential(name='Reuters')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(128, activation='relu', name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(len(ohe.categories_[0]), activation='softmax', name='Output'))
model.summary()
            
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=10, 
                 validation_split=0.2)

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

eval_result = model.evaluate(test_dataset_x, test_dataset_y, batch_size=32)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
    
 # prediction 
   
cats = ['cocoa','grain','veg-oil','earn','acq','wheat','copper','housing','money-supply',   
'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas',
'cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin',
'strategic-metal','livestock','retail','ipi','iron-steel','rubber','heat','jobs',
'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead']

import os
import re
         
word_numbers_list = []
fnames = []
for fname in os.listdir('PredictData'):
    with open('PredictData/' + fname, encoding='latin-1') as f:
        text = f.read()
        words = re.findall('[a-zA-Z0-9]+', text.lower())
        word_numbers = [vocab_dict[word] + 3 for word in words]
        word_numbers_list.append(word_numbers)
        fnames.append(fname)
    
predict_dataset_x = vectorize(word_numbers_list, len(vocab_dict) + 3)
    
predict_result = model.predict(predict_dataset_x)
predict_indexes = np.argmax(predict_result, axis=1)

for index, pi in enumerate(predict_indexes):
    print(f'{fnames[index]} => {cats[pi]}')
    
# reverse transformation test
    
rev_vocab_dict = {index: word for word, index in vocab_dict.items()}
convert_text = lambda text_numbers: ' '.join([rev_vocab_dict[tn - 3] for tn in text_numbers if tn > 2])
print(convert_text(word_numbers_list[0]))
  
    
    
