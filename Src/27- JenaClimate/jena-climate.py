import pandas as pd

PREDICTION_INTERVAL = 24 * 60 // 10         # 144

df = pd.read_csv('jena_climate_2009_2016.csv')

df['Month'] = df['Date Time'].str[3:5]  # Month sütunundaki her bir satırdaki string'in 4. ve 5. karakterini alır.
df['Hour-Minute'] = df['Date Time'].str[11:16]

df.drop(['Date Time'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['Month', 'Hour-Minute'],  dtype='int8')

dataset = df.to_numpy('float32')

dataset_x = dataset[:-PREDICTION_INTERVAL, :]
dataset_y = dataset[PREDICTION_INTERVAL, 1]


from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

from sklearn.preprocessing import StandardScaler

mms = StandardScaler()
mms.fit(training_dataset_x)
scaled_training_dataset_x = mms.transform(training_dataset_x)
scaled_test_dataset_x = mms.transform(test_dataset_x)



from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Reshape, Dense

model = Sequential(name='Jena-Climate')
model.add(Input((1, training_dataset_x.shape[1]),  name='Input'))

model.add(Conv1D(128, 3, padding='same', name='Conv1D-1'))
model.add(MaxPooling1D(2, padding='same', name='MaxPooling1D-1'))

model.add(Conv1D(128, 3, padding='same', name='Conv1D-2'))
model.add(MaxPooling1D(2, padding='same', name='MaxPooling1D-2'))

model.add(Conv1D(128, 3, padding='same', name='Conv1D-3'))
model.add(MaxPooling1D(2, padding='same', name='MaxPooling1D-3'))

model.add(Reshape((-1, ), name='Reshape'))

model.add(Dense(256, activation='relu', name='Hidden-1'))
model.add(Dense(256, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()




