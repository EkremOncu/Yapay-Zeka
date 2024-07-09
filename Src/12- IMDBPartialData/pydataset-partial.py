import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import PyDataset

EPOCHS = 2
NFEATURES = 10
BATCH_SIZE = 32

class DataGenerator(PyDataset):
    def __init__(self, batch_size, nfeatures, *, steps):
        super().__init__()
        self.batch_size = batch_size
        self.nfeatures = nfeatures
        self.steps = steps
        
    def __len__(self):
       return self.steps
    
    def __getitem__(self, batch_no):
        x = np.random.random((self.batch_size, self.nfeatures))
        y = np.random.randint(0, 2, self.batch_size)
        return x, y
    
    def on_epoch_end(self):
        print('shuffle')

model = Sequential(name='Test')

model.add(Input((NFEATURES, )))
model.add(Dense(16, activation='relu', name='Hidden-1'))
model.add(Dense(16, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(DataGenerator(BATCH_SIZE, NFEATURES, steps=50), epochs=EPOCHS, 
          validation_data=DataGenerator(BATCH_SIZE, NFEATURES, steps=5))


eval_result = model.evaluate(DataGenerator(BATCH_SIZE, NFEATURES, steps=10))
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

predict_result = model.predict(DataGenerator(BATCH_SIZE, NFEATURES, steps=1))

for presult in predict_result[:, 0]:
    if (presult > 0.5):
        print('Positive')
    else:
        print('Negative')

