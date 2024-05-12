import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# prediction

model = load_model('auto-mpg.h5')

with open('auto-mpg.pickle', 'rb') as f:
    ohe_train, ss = pickle.load(f)    

predict_df = pd.read_csv('predict.csv', header=None)

predict_df_1 = predict_df.iloc[:, :6]
predict_df_2 = predict_df.iloc[:, [6]]

predict_dataset_1 = predict_df_1.to_numpy()
predict_dataset_2 = predict_df_2.to_numpy()

predict_dataset_2  = ohe_train.transform(predict_dataset_2)

predict_dataset = np.concatenate([predict_dataset_1, predict_dataset_2], axis=1)

scaled_predict_dataset = ss.transform(predict_dataset)
predict_result = model.predict(scaled_predict_dataset)

for val in predict_result[:, 0]:
    print(val)
