import numpy as np
import pandas as pd

df = pd.read_csv('housing.csv', delimiter=r'\s+', header=None)

dataset_x = df.iloc[:, :-1].to_numpy(dtype='float32')
dataset_y = df.iloc[:, -1].to_numpy(dtype='float32')

import seaborn as sns
import matplotlib.pyplot as plt

feature_corrs = np.abs(np.corrcoef(dataset_x, rowvar=False))

plt.figure(figsize=(12, 9))
sns.heatmap(data=feature_corrs, annot=True)
plt.show()

sns.heatmap(dataset_x)
