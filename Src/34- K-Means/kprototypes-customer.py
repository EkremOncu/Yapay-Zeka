NCLUSTERS = 5

import pandas as pd

df = pd.read_csv('segmentation data.csv')
df.drop(labels=['ID'], axis=1, inplace=True)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
df[['Age', 'Income']] = ss.fit_transform(df[['Age', 'Income']])

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for column in ['Sex', 'Marital status', 'Education', 'Occupation', 'Settlement size']:
    df[column] = le.fit_transform(df[column])
    
dataset = df.to_numpy()


from kmodes.kprototypes import KPrototypes

kp = KPrototypes(n_clusters=NCLUSTERS)

kp.fit(dataset, categorical=[0, 1, 3, 5, 6])

for i in range(NCLUSTERS):
    print(f'{i}. Cluster points', end='\n\n')
    print(df.iloc[kp.labels_ == i, :])
    print('-' * 20, end='\n\n')
    

    
    