import pandas as pd
import numpy as np

df = pd.read_csv('melb_data.csv')

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean')
df[['Car', 'BuildingArea']] = np.round(si.fit_transform(df[['Car', 'BuildingArea']]))

si.set_params(strategy='median')
df[['YearBuilt']] = np.round(si.fit_transform(df[['YearBuilt']]))

si.set_params(strategy='most_frequent')
df[['CouncilArea']] = si.fit_transform(df[['CouncilArea']])


print(df.isna().sum())
print('-----------------------')


def category_encoder(df, colnames):
    for colname in colnames:
        labels = df[colname].unique()
        for index, label in enumerate(labels):
            df.loc[df[colname] == label, colname] = index


category_encoder(df, ['Suburb', 'SellerG', 'Method', 'CouncilArea', 'Regionname'])
print(df)
print(df['Suburb'])