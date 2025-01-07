import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('points.csv')
dataset = df.to_numpy(dtype='float32')

from scipy.cluster.hierarchy import linkage, dendrogram

linkage_data= linkage(dataset)

plt.title('Points Dendrogram')
dendrogram(linkage_data)
plt.show()
