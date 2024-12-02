import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('spam_ham_dataset.csv')

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(df['text'], df['label_num'])

training_dataset_x = training_dataset_x.to_numpy()
test_dataset_x = test_dataset_x.to_numpy()

training_dataset_y = training_dataset_y.to_numpy(dtype='uint8')
test_dataset_y = test_dataset_y.to_numpy(dtype='uint8')

import autokeras as ak

tc = ak.TextClassifier(max_trials=3)
hist = tc.fit(training_dataset_x, training_dataset_y, epochs=10)
