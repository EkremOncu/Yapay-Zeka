import pandas as pd

df = pd.read_csv('iris.csv')
dataset_x = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].to_numpy('float32')
dataset_y = df['Species'].to_numpy(dtype='str')

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(dataset_x)
scaled_dataset_x = ss.transform(dataset_x)

from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier(5)
knc.fit(scaled_dataset_x, dataset_y)

predict_df = pd.read_csv('predict-iris.csv')
predict_dataset = predict_df.to_numpy(dtype='float32')
scaled_dataset_predict = ss.transform(predict_dataset)

predict_result = knc.predict(scaled_dataset_predict)
print(predict_result)



