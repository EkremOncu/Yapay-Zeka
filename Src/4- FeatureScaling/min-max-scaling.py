import numpy as np

def minmax_scaler(dataset):
    scaled_dataset = np.zeros(dataset.shape)
    
    for col in range(dataset.shape[1]):
        min_val, max_val = np.min(dataset[:, col]), np.max(dataset[:, col])
        scaled_dataset[:, col] = 0 if max_val - min_val == 0 else (dataset[:, col] - min_val) / (max_val - min_val)
        
    return scaled_dataset

"""
def minmax_scaler(dataset):
        return (dataset - np.min(dataset, axis=0)) / (np.max(dataset, axis=0) - np.min(dataset, axis=0))
"""

dataset = np.array([[1, 2, 3], [2, 1, 4], [7, 3, 8], [8, 9, 2], [20, 12, 3]])

result = minmax_scaler(dataset)
print(dataset)
print()
print(result)

print('------------------------------------------------')

dataset = np.array([[1, 2, 3], [2, 1, 4], [7, 3, 8], [8, 9, 2], [20, 12, 3]])

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
mms.fit(dataset)
scaled_dataset = mms.transform(dataset)

print(dataset)
print()
print(scaled_dataset)





