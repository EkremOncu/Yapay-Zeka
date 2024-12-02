import pandas as pd

df = pd.read_csv('Titanic-Dataset.csv')

dataset_y = df['Survived'].to_numpy('uint8')

df = df.drop(['Survived', 'PassengerId', 'Cabin', 'Name', 'Parch', 'Ticket'], axis=1)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean')
df['Age'] = si.fit_transform(df[['Age']])

dataset_x = df.to_numpy('float32')

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, 
                                                                                          dataset_y, test_size=0.2)


import autokeras as ak

inp = ak.Input()
x = ak.DenseBlock()(inp)
out = ak.ClassificationHead()(x)
auto_model = ak.AutoModel(inputs=inp, outputs=out, max_trials=100, overwrite=True)

hist = auto_model.fit(training_dataset_x, training_dataset_y, validation_split=0.2, epochs=50)

keras_model = auto_model.export_model()
keras_model.summary()



import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.title('Epoch-Loss Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.plot(hist.epoch, hist.history['loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(15, 5))
plt.title('Epoch-Binary Accuracy Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.plot(hist.epoch, hist.history['accuracy'])
plt.legend(['Binary Accuracy', 'Validation Binary Accuracy'])
plt.show()

# evaluation

eval_result = auto_model.evaluate(test_dataset_x, test_dataset_y)
for i in range(len(eval_result)):
    print(f'{keras_model.metrics_names[i]}: {eval_result[i]}')

# prediction

df_predict = pd.read_csv('predict.csv')

df_predict = df_predict.drop(['PassengerId', 'Cabin', 'Name', 'Parch', 'Ticket'], axis=1)

le = LabelEncoder()

df_predict['Sex'] = le.fit_transform(df_predict['Sex'])
df_predict['Embarked'] = le.fit_transform(df_predict['Embarked'])

predict_dataset_x = df_predict.to_numpy('float32')
predict_result = auto_model.predict(predict_dataset_x)
print(predict_result)

