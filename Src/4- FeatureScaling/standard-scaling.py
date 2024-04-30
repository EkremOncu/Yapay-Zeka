import numpy as np

dataset = np.array([[1, 2, 3], [2, 1, 4], [7, 3, 8], [8, 9, 2], [20, 12, 3]])

def standard_scaler(dataset):
    scaled_dataset = np.zeros(dataset.shape)
    for col in range(dataset.shape[1]):
        scaled_dataset[:, col] = (dataset[:, col] - np.mean(dataset[:, col])) / np.std(dataset[:, col])
    return scaled_dataset

"""
def standard_scaler(dataset):
     return (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)
"""

result = standard_scaler(dataset)     
print(dataset)
print()
print(result)
    
print('--------------------------------')

dataset = np.array([[1, 2, 3], [2, 1, 4], [7, 3, 8], [8, 9, 2], [20, 12, 3]])
print(dataset)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(dataset)

print(f'{ss.mean_}, {ss.scale_}')
scaled_dataset = ss.transform(dataset)
print()
print(scaled_dataset)











 
    