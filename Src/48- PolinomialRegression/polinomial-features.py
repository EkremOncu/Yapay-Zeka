import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

dataset_x = pd.read_csv('points.csv')
pf = PolynomialFeatures(2)
transformed_dataset_x = pf.fit_transform(dataset_x)
print(transformed_dataset_x )

