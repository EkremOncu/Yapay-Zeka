import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

EPOCHS = 100
NFEATURES = 10
STEPS_PER_EPOCH = 20
BATCH_SIZE = 32
VALIDATION_STEPS = 10
EVALUATION_STEPS = 15
PREDICTION_STEPS = 5

def data_generator():
    for _ in range(EPOCHS):
        for _ in range(STEPS_PER_EPOCH):
            x = np.random.random((BATCH_SIZE, NFEATURES))
            y = np.random.randint(0, 2, BATCH_SIZE)
            yield x, y   

def validation_generator():
    x = np.random.random((BATCH_SIZE, NFEATURES))
    y = np.random.randint(0, 2, BATCH_SIZE)
    for _ in range(EPOCHS):
        for _ in range(VALIDATION_STEPS + 1):
            yield x, y

def evaluation_generator():
    for _ in range(EVALUATION_STEPS):
        x = np.random.random((BATCH_SIZE, NFEATURES))
        y = np.random.randint(0, 2, BATCH_SIZE)
        yield x, y
        
def prediction_generator():
    for _ in range(PREDICTION_STEPS):
        x = np.random.random((BATCH_SIZE, NFEATURES))
        yield x

model = Sequential(name='Diabetes')

model.add(Input(NFEATURES, ))
model.add(Dense(16, activation='relu', name='Hidden-1'))
model.add(Dense(16, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', 
              metrics=['binary_accuracy'])

model.fit(data_generator(), epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, 
          validation_data=validation_generator(), validation_steps=VALIDATION_STEPS)

eval_result = model.evaluate(evaluation_generator(), steps=EVALUATION_STEPS)

predict_result = model.predict(prediction_generator(), steps=PREDICTION_STEPS)



