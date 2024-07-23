import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

dense = np.zeros((10, 5))

for i in range(len(dense)):
    rcols = np.random.randint(0, 5, 2)
    dense[i, rcols] = np.random.randint(0, 100, 2)
    
sparse_dataset_x = csr_matrix(dense)
dataset_y = np.random.randint(0, 2, 10)

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(sparse_dataset_x, 
                                                                                dataset_y, test_size=0.2)

print(training_dataset_x)
print('-' * 20)
print(training_dataset_y)
print()
print()

print(test_dataset_x)
print('-' * 20)
print(test_dataset_y)

