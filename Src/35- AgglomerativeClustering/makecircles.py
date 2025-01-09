from sklearn.datasets import make_circles

dataset, labels = make_circles(100, factor=0.8, noise=0.05)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.title('Clustered Points')

for i in range(2):
    plt.scatter(dataset[labels == i, 0], dataset[labels == i, 1])    
plt.show()
