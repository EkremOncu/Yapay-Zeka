import pandas as pd


df = pd.read_csv('IMDB Dataset.csv')


dataset_x = df['review'].to_numpy()   
dataset_y = df['sentiment'].to_numpy()


from sklearn.model_selection import train_test_split


training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y)


import autokeras as ak

tc = ak.TextClassifier(max_trials=3, overwrite=True)
hist = tc.fit(training_dataset_x, training_dataset_y, epochs=10)



import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.title('Epoch-Loss Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')


plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()


plt.figure(figsize=(15, 5))
plt.title('Epoch-Binary Accuracy Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')


plt.plot(hist.epoch, hist.history['binary_accuracy'])
plt.plot(hist.epoch, hist.history['val_binary_accuracy'])
plt.legend(['Binary Accuracy', 'Validation Binary Accuracy'])
plt.show()


model = tc.export_model()
model.summary()
model.save('text-classifier-imdb-best-model.h5')


eval_result = tc.evaluate(test_dataset_x, test_dataset_y)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')


texts = ['the movie was very good. The actors played perfectly. I would recommend it to everyone.', 
        'this film is awful. The worst film i have ever seen']


for predict_text in texts:
    predict_result = tc.predict(texts)  
    if predict_result[0, 0] > 0.5:
        print('Positive')
    else:
        print('Negative')
        
model.save('imdb.h5')
