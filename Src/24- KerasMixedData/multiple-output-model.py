TOTAL_ITEM = 1000
NFEATURES = 10

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense 

inp = Input(shape=(NFEATURES, ), name='Input')
result = Dense(64, activation='relu', name='Dense-1')(inp)
result = Dense(64, activation='relu', name='Dense-2')(result)
out1 = Dense(1, activation='sigmoid', name='Output-1')(result)
out2 = Dense(1, activation='linear', name='Output-2')(result)

model = Model(inputs=inp, outputs=[out1, out2])

model.summary()

model.compile(optimizer='rmsprop', loss={'Output-1': 'binary_crossentropy', 'Output-2': 'mse'}, 
              loss_weights={'Output-1': 800.0, 'Output-2': 1.0}, 
              metrics={'Output-1': ['binary_accuracy'], 'Output-2': ['mse']})

# generate random data

import numpy as np

dataset_x = np.random.random((TOTAL_ITEM, NFEATURES))
dataset_y1 = np.random.randint(0, 2, TOTAL_ITEM)
dataset_y2 = np.random.randint(0, 100, TOTAL_ITEM).astype('float32')

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y1, test_dataset_y1, training_dataset_y2, test_dataset_y2 = train_test_split(dataset_x, dataset_y1, dataset_y2, test_size = 0.2)

hist = model.fit(training_dataset_x, [training_dataset_y1, training_dataset_y2], 
                 batch_size=32, epochs=100, validation_split=0.2)

eval_result = model.evaluate(test_dataset_x, [test_dataset_y1, test_dataset_y2])
print(eval_result)

# generate random data for prediction 

predict_data = np.random.random(NFEATURES)
predict_result = model.predict(predict_data.reshape(1, -1))

print(predict_result[0][0, 0], predict_result[1][0, 0])

