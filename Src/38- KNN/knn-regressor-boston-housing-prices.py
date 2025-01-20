import pandas as pd

df = pd.read_csv('housing.csv', delimiter=r'\s+', header=None)

dataset_y = df.iloc[:, -1].to_numpy()
df.drop([8, 13], axis=1, inplace=True)
dataset_x = df.to_numpy('float32')

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(dataset_x)
scaled_dataset_x = ss.transform(dataset_x)

from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(5)
knr.fit(scaled_dataset_x, dataset_x)

predict_df = pd.read_csv('predict-boston-housing-prices.csv', delimiter=r'\s+', header=None)

predict_df.drop([8], axis=1, inplace=True)
predict_dataset_x = predict_df.to_numpy('float32')

scaled_predict_dataset_x = ss.transform(predict_dataset_x)

predict_result = knr.predict(scaled_predict_dataset_x)
print(predict_result)


    


















